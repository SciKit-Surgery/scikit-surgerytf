# -*- coding: utf-8 -*-

"""
Module to implement a semantic (pixelwise) segmentation of images of the liver.
"""
import os
import sys
import glob
import logging
import datetime
import platform
import random
import ssl
import shutil
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import cv2
from sksurgerytf import __version__

LOGGER = logging.getLogger(__name__)


class LiverSeg:
    """
    Class to encapsulate LiverSeg semantic (pixelwise) segmentation network.

    Thanks to
    `Harshall Lamba <https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47>_,
    `Zhixuhao <https://github.com/zhixuhao/unet/blob/master/model.py>`_
    and
    `ShawDa <https://github.com/ShawDa/unet-rgb/blob/master/unet.py>`_
    for inspiration.
    """
    def __init__(self,
                 logs="logs/fit",
                 data=None,
                 working=None,
                 omit=None,
                 model=None,
                 learning_rate=0.001,
                 epochs=3,
                 batch_size=4,
                 input_size=(512, 512, 3)
                 ):
        """
        Class to implement a CNN to extract a binary mask
        of the liver from RGB video.

        If the constructor is called without weights, the data is loaded
        and a full training cycle is performed in order to learn the weights.

        If the constructor is called with weights, these weights are loaded,
        as is, with no further training. If you want to continue training, call
        the train method again.

        :param logs: relative path to folder to write tensorboard log files.
        :param data: root directory of training data.
        :param working: working directory for organising data.
        :param omit: patient identifier to omit, when doing Leave-One-Out.
        :param model: file name of previously saved model.
        :param learning_rate: float, default=0.001 which is the Keras default.
        :param epochs: int, default=1,
        :param batch_size: int, default=32,
        :param input_size: Expected input size for network.
        """
        self.logs = logs
        self.data = data
        self.working = working
        self.omit = omit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size

        self.model = None
        self.train_images_working_dir = None
        self.train_masks_working_dir = None
        self.train_generator = None
        self.number_training_samples = None
        self.validate_images_working_dir = None
        self.validate_masks_working_dir = None
        self.validate_generator = None
        self.number_validation_samples = None

        LOGGER.info("Creating LiverSeg with log dir: %s.",
                    str(self.logs))
        LOGGER.info("Creating LiverSeg with model file: %s.",
                    str(model))
        LOGGER.info("Creating LiverSeg with data dir: %s.",
                    str(self.data))
        LOGGER.info("Creating LiverSeg with working dir: %s.",
                    str(self.working))
        LOGGER.info("Creating LiverSeg with omit: %s.",
                    str(self.omit))
        LOGGER.info("Creating LiverSeg with learning_rate: %s.",
                    str(self.learning_rate))
        LOGGER.info("Creating LiverSeg with epochs: %s.",
                    str(self.epochs))
        LOGGER.info("Creating LiverSeg with batch_size: %s.",
                    str(self.batch_size))
        LOGGER.info("Creating LiverSeg with input_size size: %s.",
                    str(self.input_size))

        # To fix issues with SSL certificates on CI servers.
        ssl._create_default_https_context = ssl._create_unverified_context

        if model is None:

            if self.working is None:
                raise ValueError("You must specify a working (temp) directory")
            if self.data is None:
                raise ValueError("You must specify the data directory")

            self._copy_data()
            self._load_data()
            self._build_model()
            self.train()

        else:
            self.model = keras.models.load_model(model)

    def _copy_data(self):
        """
        Copies data from data directory to working directory.

        If the user is doing 'Leave-On-Out' then we validate on that case.
        If the user hasn't specified a L-O-O, we pick one at random.
        """
        # Look for each case in a sub-directory.
        sub_dirs = [f.path for f in os.scandir(self.data) if f.is_dir()]
        if not sub_dirs:
            raise ValueError("Couldn't find sub directories")

        # If the user hasn't specified a L-O-O, we pick one at random.
        if self.omit is None:
            self.omit = os.path.basename(
                sub_dirs[random.randint(0, len(sub_dirs))])
            LOGGER.info("Chose random validation set:%s", self.omit)

        # Always recreate working directory to avoid data leak.
        if os.path.exists(self.working):
            LOGGER.info("Removing working directory: %s", self.working)
            shutil.rmtree(self.working)

        # Keras still requires a sub-dir for the class name.
        class_name = 'liver'

        self.train_images_working_dir = os.path.join(self.working,
                                                     'train',
                                                     'images',
                                                     class_name)
        LOGGER.info("Creating directory: %s", self.train_images_working_dir)
        os.makedirs(self.train_images_working_dir)

        self.train_masks_working_dir = os.path.join(self.working,
                                                    'train',
                                                    'masks',
                                                    class_name)
        LOGGER.info("Creating directory: %s", self.train_masks_working_dir)
        os.makedirs(self.train_masks_working_dir)

        self.validate_images_working_dir = os.path.join(self.working,
                                                        'validate',
                                                        'images',
                                                        class_name)
        LOGGER.info("Creating directory: %s", self.validate_images_working_dir)
        os.makedirs(self.validate_images_working_dir)

        self.validate_masks_working_dir = os.path.join(self.working,
                                                       'validate',
                                                       'masks',
                                                       class_name)
        LOGGER.info("Creating directory: %s", self.validate_masks_working_dir)
        os.makedirs(self.validate_masks_working_dir)

        for sub_dir in sub_dirs:

            images_sub_dir = os.path.join(sub_dir, 'images')
            mask_sub_dir = os.path.join(sub_dir, 'masks')

            if self.omit is not None and self.omit == os.path.basename(sub_dir):
                LOGGER.info("Copying validate images from %s to %s",
                            images_sub_dir, self.validate_images_working_dir)
                for image_file in glob.iglob(
                        os.path.join(images_sub_dir, "*.png")):
                    shutil.copy(image_file, self.validate_images_working_dir)

                LOGGER.info("Copying validate masks from %s to %s",
                            mask_sub_dir, self.validate_masks_working_dir)
                for mask_file in glob.iglob(
                        os.path.join(mask_sub_dir, "*.png")):
                    shutil.copy(mask_file, self.validate_masks_working_dir)
            else:
                LOGGER.info("Copying train images from %s to %s",
                            images_sub_dir, self.train_images_working_dir)
                for image_file in glob.iglob(
                        os.path.join(images_sub_dir, "*.png")):
                    shutil.copy(image_file, self.train_images_working_dir)

                LOGGER.info("Copying train masks from %s to %s",
                            mask_sub_dir, self.train_masks_working_dir)
                for mask_file in glob.iglob(
                        os.path.join(mask_sub_dir, "*.png")):
                    shutil.copy(mask_file, self.train_masks_working_dir)

    def _load_data(self):
        """
        Sets up the Keras ImageDataGenerator to load images and masks together.
        """
        train_data_gen_args = dict(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=0,
                                   width_shift_range=0,
                                   height_shift_range=0,
                                   zoom_range=0)

        validate_data_gen_args = dict(rescale=1./255)

        train_image_datagen = keras.preprocessing.image.ImageDataGenerator(
            **train_data_gen_args)
        train_mask_datagen = keras.preprocessing.image.ImageDataGenerator(
            **train_data_gen_args)

        validate_image_datagen = keras.preprocessing.image.ImageDataGenerator(
            **validate_data_gen_args)
        validate_mask_datagen = keras.preprocessing.image.ImageDataGenerator(
            **validate_data_gen_args)

        seed = 1

        train_image_generator = train_image_datagen.flow_from_directory(
            os.path.dirname(self.train_images_working_dir),
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode=None,
            shuffle=True,
            seed=seed)

        train_mask_generator = train_mask_datagen.flow_from_directory(
            os.path.dirname(self.train_masks_working_dir),
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode=None,
            shuffle=True,
            seed=seed)

        self.number_training_samples = len(train_image_generator.filepaths)

        self.train_generator = (pair for pair in zip(train_image_generator, train_mask_generator))

        validate_image_generator = validate_image_datagen.flow_from_directory(
            os.path.dirname(self.validate_images_working_dir),
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode=None,
            shuffle=False,
            seed=seed)

        validate_mask_generator = validate_mask_datagen.flow_from_directory(
            os.path.dirname(self.validate_masks_working_dir),
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode=None,
            shuffle=False,
            seed=seed)

        self.number_validation_samples = len(validate_image_generator.filepaths)

        self.train_generator = (pair for pair in zip(validate_image_generator, validate_mask_generator))

    def _build_model(self):
        """
        Constructs the neural network, and compiles it.

        Currently, we are using a standard UNet on RGB images.
        """

        LOGGER.info("Building Model")

        inputs = keras.Input(self.input_size)

        # Left side of UNet
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottom of UNet
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        # Right side of UNet
        up6 = keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv5))
        merge6 = keras.layers.concatenate([conv4, up6])
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = keras.layers.concatenate([conv3, up7])
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = keras.layers.concatenate([conv2, up8])
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        up9 = keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = keras.layers.concatenate([conv1, up9])
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

        self.model = keras.models.Model(inputs=inputs, outputs=conv10)
        self.model.summary()

        LOGGER.info("Built Model")

    def train(self):
        """
        Method to train the neural network. Writes each epoch
        to tensorboard log files.

        :return: output of self.model.evaluate on test set.
        """

        LOGGER.info("Training Model")

        optimiser = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=optimiser,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        log_dir = os.path.join(Path(self.logs),
                               datetime.datetime.now()
                               .strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=1)

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.number_training_samples // self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.validate_generator,
            validation_steps=self.number_validation_samples // self.batch_size,
            callbacks=[tensorboard_callback]
        )

        return self.model.evaluate(self.validate_generator,
                                   batch_size=self.batch_size,
                                   steps=self.number_validation_samples // self.batch_size,
                                   verbose=2
                                   )

    def test(self, image):
        """
        Method to test a single (1920 x 540) image.

        :param image: (1920 x 540), numpy, 3 channel RGB, [0-255], uchar.
        :return: (1920 x 540) numpy, single channel, [0=background|255=liver].
        """
        return np.zeros((540, 1920))

    def save_model(self, filename):
        """
        Method to save the whole trained network to disk.

        :param filename: file to save to.
        """
        self.model.save(filename)


def run_liverseg_model(logs,
                       data,
                       working,
                       omit,
                       model,
                       save,
                       test):
    """
    Helper function to run the LiverSeg model from
    the command line entry point.

    :param logs: directory for log files for tensorboard.
    :param data: root directory of training data.
    :param working: working directory for organising data.
    :param omit: patient identifier to omit, when doing Leave-One-Out.
    :param model: file of previously saved model.
    :param save: file to save model to.
    :param test: image to test.
    """
    # pylint: disable=line-too-long
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    LOGGER.info("Starting liverseg.py version: %s", __version__)
    LOGGER.info("Starting liverseg.py with platform: %s.", str(platform.uname()))
    LOGGER.info("Starting liverseg.py with cwd: %s.", os.getcwd())
    LOGGER.info("Starting liverseg.py with path: %s.", sys.path)

    ls = LiverSeg(logs, data, working, omit, model)

    if save is not None:
        ls.save_model(save)

    if test is not None:
        img = cv2.imread(test)
        mask = ls.test(img)
        cv2.imwrite(test + ".mask.png", mask)
