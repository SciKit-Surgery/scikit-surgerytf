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
import cv2


from sksurgerytf import __version__

LOGGER = logging.getLogger(__name__)


class LiverSeg:
    """
    Class to encapsulate LiverSeg semantic (pixelwise) segmentation network.
    """
    def __init__(self,
                 logs="logs/fit",
                 data=None,
                 working=None,
                 omit=None,
                 model=None,
                 learning_rate=0.001,
                 epochs=1
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
        :param epochs: int, default=1
        """
        self.logs = logs
        self.data = data
        self.working = working
        self.omit = omit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.train_images_working_dir = None
        self.train_masks_working_dir = None
        self.train_generator = None
        self.validate_images_working_dir = None
        self.validate_masks_working_dir = None
        self.validate_generator = None
        self.image_size = (1920, 540)
        self.input_size = (256, 256)

        LOGGER.info("Creating LiverSeg with log dir: %s.",
                    str(self.logs))
        LOGGER.info("Creating LiverSeg with data dir: %s.",
                    str(self.data))
        LOGGER.info("Creating LiverSeg with working dir: %s.",
                    str(self.working))
        LOGGER.info("Creating LiverSeg with omit: %s.",
                    str(self.omit))
        LOGGER.info("Creating LiverSeg with model file: %s.",
                    str(model))
        LOGGER.info("Creating LiverSeg with learning_rate: %s.",
                    str(self.learning_rate))
        LOGGER.info("Creating LiverSeg with epochs: %s.",
                    str(self.epochs))

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
            target_size=self.input_size,
            batch_size=32,
            color_mode='rgb',
            class_mode=None,
            shuffle=True,
            seed=seed)

        train_mask_generator = train_mask_datagen.flow_from_directory(
            os.path.dirname(self.train_masks_working_dir),
            target_size=self.input_size,
            batch_size=32,
            color_mode='grayscale',
            class_mode=None,
            shuffle=True,
            seed=seed)

        self.train_generator = zip(train_image_generator,
                                   train_mask_generator)

        validate_image_generator = validate_image_datagen.flow_from_directory(
            os.path.dirname(self.validate_images_working_dir),
            target_size=self.input_size,
            batch_size=32,
            color_mode='rgb',
            class_mode=None,
            shuffle=False,
            seed=seed)

        validate_mask_generator = validate_mask_datagen.flow_from_directory(
            os.path.dirname(self.validate_masks_working_dir),
            target_size=self.input_size,
            batch_size=32,
            color_mode='grayscale',
            class_mode=None,
            shuffle=False,
            seed=seed)

        self.validate_generator = zip(validate_image_generator,
                                      validate_mask_generator)

    def _build_model(self):
        """
        Constructs the neural network, and compiles it.

        Currently, we are using a standard UNet.

        Remember that the Liver cases are pre-cropped and hence may be of
        different sizes, so they will need resizing as input to network.
        """

        LOGGER.info("Building Model")

        #optimiser = keras.optimizers.Adam(learning_rate=self.learning_rate)

        #model.fit_generator(
        #    train_generator,
        #    steps_per_epoch=2000,
        #    epochs=50)

        #self.model.compile(optimizer=optimiser,
        #                   loss='sparse_categorical_crossentropy',
        #                   metrics=['accuracy'])

        #self.model.summary()

    def train(self):
        """
        Method to train the neural network. Writes each epoch
        to tensorboard log files.

        :return: output of self.model.evaluate on test set.
        """

        LOGGER.info("Training Model")

        #log_dir = os.path.join(Path(self.logs),
        #                       datetime.datetime.now()
        #                       .strftime("%Y%m%d-%H%M%S"))
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
        #                                                   histogram_freq=1)

        #self.model.fit(self.train_images,
        #               self.train_labels,
        #               epochs=self.epochs,
        #               validation_data=(self.test_images, self.test_labels),
        #               callbacks=[tensorboard_callback]
        #               )

        #return self.model.evaluate(self.test_images,
        #                           self.test_labels,
        #                           verbose=2,
        #                           )

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
