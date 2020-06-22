# -*- coding: utf-8 -*-

"""
Module to implement a semantic (pixelwise) segmentation using UNet on 512x512.
"""

#pylint: disable=line-too-long, too-many-instance-attributes, unsubscriptable-object, too-many-branches, too-many-arguments

import os
import sys
import glob
import logging
import datetime
import platform
import ssl
import shutil
import getpass
from pathlib import Path
import numpy as np
import cv2
from tensorflow import keras
from sksurgerytf import __version__
import sksurgerytf.callbacks.segmentation_history as sh
import sksurgerytf.utils.segmentation_statistics as ss

LOGGER = logging.getLogger(__name__)


class RGBUNet:
    """
    Class to encapsulate RGB UNet semantic (pixelwise) segmentation network.

    Thanks to
    `Zhixuhao <https://github.com/zhixuhao/unet/blob/master/model.py>`_,
    and
    `ShawDa <https://github.com/ShawDa/unet-rgb/blob/master/unet.py>`_
    for getting me started, and
    `Harshall Lamba <https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47>_,
    for further inspiration.
    """
    def __init__(self,
                 logs="logs/fit",
                 data=None,
                 working=None,
                 omit=None,
                 model=None,
                 learning_rate=0.0001,
                 epochs=50,
                 batch_size=2,
                 input_size=(512, 512, 3),
                 patience=20
                 ):
        """
        Class to run UNet on RGB images.

        If the constructor is called without a previously saved model,
        the data is loaded and a full training cycle is performed.

        If the constructor is called with a previously saved model,
        the model is loaded as is, with no further training. You can then
        call the test method to predict the output on new images.

        :param logs: relative path to folder to write tensorboard log files.
        :param data: root directory of training data.
        :param working: working directory for organising data.
        :param omit: patient identifier to omit, when doing Leave-One-Out.
        :param model: file name of previously saved model.
        :param learning_rate: float, default=0.001 for Adam optimiser.
        :param epochs: int, default=3,
        :param batch_size: int, default=4,
        :param input_size: Expected input size for network, default (512,512,3).
        :param patience: number of steps to tolerate non-improving accuracy
        """
        LOGGER.info("Creating RGBUNet with log dir: %s.",
                    str(logs))
        LOGGER.info("Creating RGBUNet with data dir: %s.",
                    str(data))
        LOGGER.info("Creating RGBUNet with working dir: %s.",
                    str(working))
        LOGGER.info("Creating RGBUNet with omit: %s.",
                    str(omit))
        LOGGER.info("Creating RGBUNet with model file: %s.",
                    str(model))
        LOGGER.info("Creating RGBUNet with learning_rate: %s.",
                    str(learning_rate))
        LOGGER.info("Creating RGBUNet with epochs: %s.",
                    str(epochs))
        LOGGER.info("Creating RGBUNet with batch_size: %s.",
                    str(batch_size))
        LOGGER.info("Creating RGBUNet with input_size size: %s.",
                    str(input_size))
        LOGGER.info("Creating RGBUNet with patience: %s.",
                    str(patience))

        self.logs = logs
        self.data = data
        self.working = working
        self.omit = omit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.patience = patience

        self.model = None
        self.train_images_working_dir = None
        self.train_masks_working_dir = None
        self.train_generator = None
        self.number_training_samples = None
        self.validate_images_working_dir = None
        self.validate_masks_working_dir = None
        self.validate_generator = None
        self.number_validation_samples = None

        # To fix issues with SSL certificates on CI servers.
        ssl._create_default_https_context = ssl._create_unverified_context

        if model is not None:
            LOGGER.info("Loading Model")
            self.model = keras.models.load_model(model)
            LOGGER.info("Loaded Model")
        else:
            LOGGER.info("Building Model")
            self.model = self._build_model()
            LOGGER.info("Built Model")
        self.model.summary()

        if model is None and self.working is None:
            raise ValueError("You must specify a working (temp) directory")
        if model is None and self.data is None:
            raise ValueError("You must specify the data directory")

        if self.data is not None and self.working is not None:
            self._copy_data()
            self._load_data()
            self.train()

    def _copy_images(self, src_dir, dst_dir):
        """
        Symlinks .png files from one directory to another.
        """
        #pylint: disable=no-self-use
        for image_file in glob.iglob(os.path.join(src_dir, "*.png")):
            destination = os.path.join(dst_dir,
                                       os.path.basename(
                                           os.path.dirname(src_dir)) + "_" +
                                       os.path.basename(image_file))
            os.symlink(image_file, destination)

    def _copy_data(self):
        """
        Copies data from data directory to working directory.

        If the user is doing 'Leave-On-Out' then we validate on that case.
        """
        # Look for each case in a sub-directory.
        sub_dirs = [f.path for f in os.scandir(self.data) if f.is_dir()]
        if not sub_dirs:
            raise ValueError("Couldn't find sub directories")
        sub_dirs.sort()

        if self.omit is not None:
            found_it = False
            for directory in sub_dirs:
                if os.path.basename(directory) == self.omit:
                    found_it = True
                    break
            if not found_it:
                raise ValueError("User requested to omit:" +
                                 self.omit + ", but it cannot be found in:" +
                                 self.data)

        # Always recreate working directory to avoid data leak.
        if os.path.exists(self.working):
            LOGGER.info("Removing working directory: %s", self.working)
            shutil.rmtree(self.working)

        # Keras still requires a sub-dir for the class name.
        class_name = 'object'

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
                LOGGER.info("Sym-linking validate images from %s to %s",
                            images_sub_dir, self.validate_images_working_dir)
                self._copy_images(images_sub_dir,
                                  self.validate_images_working_dir)

                LOGGER.info("Sym-linking validate masks from %s to %s",
                            mask_sub_dir, self.validate_masks_working_dir)
                self._copy_images(mask_sub_dir, self.validate_masks_working_dir)
            else:
                LOGGER.info("Sym-linking train images from %s to %s",
                            images_sub_dir, self.train_images_working_dir)
                self._copy_images(images_sub_dir, self.train_images_working_dir)

                LOGGER.info("Sym-linking train masks from %s to %s",
                            mask_sub_dir, self.train_masks_working_dir)
                self._copy_images(mask_sub_dir, self.train_masks_working_dir)

    def _load_data(self):
        """
        Sets up the Keras ImageDataGenerator to load images and masks together.
        """
        train_data_gen_args = dict(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   fill_mode='nearest',
                                   rotation_range=10,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   zoom_range=[0.9, 1.0]
                                   )

        # This is for validation. We don't want data augmentation.
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

        self.train_generator = zip(train_image_generator, train_mask_generator)

        if self.omit is not None:
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

            self.validate_generator = zip(validate_image_generator, validate_mask_generator)

    # pylint: disable=no-self-use
    def _create_2d_block(self, input_tensor, num_filters, kernel_size, batch_norm=True):

        model = keras.layers.Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(input_tensor)

        if batch_norm:
            model = keras.layers.BatchNormalization()(model)

        model = keras.layers.Activation('relu')(model)

        model = keras.layers.Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(model)

        if batch_norm:
            model = keras.layers.BatchNormalization()(model)

        model = keras.layers.Activation('relu')(model)

        return model

    def _build_model(self):
        """
        Constructs the neural network, and compiles it.

        Currently, we are using a standard UNet on RGB images.
        """
        dropout = 0.1
        batch_norm = True
        kernel_size = 3
        pooling_size = 2
        num_filters = 64

        inputs = keras.Input(self.input_size)

        # Left side of UNet
        conv1 = self._create_2d_block(inputs, num_filters * 1, kernel_size=kernel_size, batch_norm=batch_norm)
        pool1 = keras.layers.MaxPooling2D((pooling_size, pooling_size))(conv1)
        pool1 = keras.layers.Dropout(dropout)(pool1)

        conv2 = self._create_2d_block(pool1, num_filters * 2, kernel_size=kernel_size, batch_norm=batch_norm)
        pool2 = keras.layers.MaxPooling2D((pooling_size, pooling_size))(conv2)
        pool2 = keras.layers.Dropout(dropout)(pool2)

        conv3 = self._create_2d_block(pool2, num_filters * 4, kernel_size=kernel_size, batch_norm=batch_norm)
        pool3 = keras.layers.MaxPooling2D((pooling_size, pooling_size))(conv3)
        pool3 = keras.layers.Dropout(dropout)(pool3)

        conv4 = self._create_2d_block(pool3, num_filters * 8, kernel_size=kernel_size, batch_norm=batch_norm)
        pool4 = keras.layers.MaxPooling2D((pooling_size, pooling_size))(conv4)
        pool4 = keras.layers.Dropout(dropout)(pool4)

        # Bottom of UNet
        conv5 = self._create_2d_block(pool4, num_filters * 16, kernel_size=kernel_size, batch_norm=batch_norm)

        # Right side of UNet
        up6 = keras.layers.Conv2DTranspose(num_filters * 8, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
        up6 = keras.layers.concatenate([up6, conv4])
        up6 = keras.layers.Dropout(dropout)(up6)
        conv6 = self._create_2d_block(up6, num_filters * 8, kernel_size=3, batch_norm=batch_norm)

        up7 = keras.layers.Conv2DTranspose(num_filters * 4, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
        up7 = keras.layers.concatenate([up7, conv3])
        up7 = keras.layers.Dropout(dropout)(up7)
        conv7 = self._create_2d_block(up7, num_filters * 4, kernel_size=3, batch_norm=batch_norm)

        up8 = keras.layers.Conv2DTranspose(num_filters * 2, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
        up8 = keras.layers.concatenate([up8, conv2])
        up8 = keras.layers.Dropout(dropout)(up8)
        conv8 = self._create_2d_block(up8, num_filters * 2, kernel_size=3, batch_norm=batch_norm)

        up9 = keras.layers.Conv2DTranspose(num_filters * 1, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
        up9 = keras.layers.concatenate([up9, conv1])
        up9 = keras.layers.Dropout(dropout)(up9)
        conv9 = self._create_2d_block(up9, num_filters * 1, kernel_size=3, batch_norm=batch_norm)

        conv10 = keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(conv9)

        return keras.models.Model(inputs=inputs, outputs=conv10)

    def train(self):
        """
        Method to train the neural network. Writes each epoch
        to tensorboard log files.

        :return: output of self.model.evaluate on validation set, or None.
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

        if self.omit is not None:
            checkpoint_filename = "checkpoint-" + self.omit + ".hdf5"
            monitor = 'val_accuracy'
        else:
            checkpoint_filename = "checkpoint-all.hdf5"
            monitor = 'accuracy'

        filepath = os.path.join(Path(self.logs),
                                checkpoint_filename)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                     monitor=monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max')

        early_stopping = keras.callbacks.EarlyStopping(monitor=monitor,
                                                       patience=self.patience,
                                                       restore_best_weights=True
                                                       )

        validation_steps = None
        if self.number_validation_samples is not None:
            validation_steps = self.number_validation_samples // self.batch_size
            segmentation_history = sh.SegmentationHistory(tensor_board_dir=log_dir,
                                                          data=self.validate_generator,
                                                          number_of_samples=self.number_validation_samples,
                                                          desired_number_images=10)
        else:
            segmentation_history = sh.SegmentationHistory(tensor_board_dir=log_dir,
                                                          data=self.train_generator,
                                                          number_of_samples=self.number_training_samples,
                                                          desired_number_images=10)

        # Note: EarlyStopping must be last. See https://github.com/keras-team/keras/issues/13381
        callbacks_list = [tensorboard_callback, segmentation_history, checkpoint, early_stopping]

        LOGGER.info("Training. Train set=%s images, batch size=%s number of batches=%s",
                    str(self.number_training_samples),
                    str(self.batch_size),
                    str(self.number_training_samples // self.batch_size))

        if self.number_validation_samples is not None:
            LOGGER.info("Training. Validation set=%s images, batch size=%s number of batches=%s",
                        str(self.number_validation_samples),
                        str(self.batch_size),
                        str(self.number_validation_samples // self.batch_size))

        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.number_training_samples // self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=self.validate_generator,  # this will be None if you didn't specify self.omit
            validation_steps=validation_steps,        # and then this won't matter if the above is None.
            callbacks=callbacks_list
        )

        result = None
        if self.validate_generator is not None and self.number_validation_samples is not None:
            result = self.model.evaluate(self.validate_generator,
                                         steps=self.number_validation_samples,
                                         verbose=2
                                         )
        return result

    def predict(self, rgb_image):
        """
        Method to test a single image. Image resized to match network,
        segmented and then resized back to match the input size.

        :param rgb_image: 3 channel RGB, [0-255], uchar.
        :return: single channel, [0=bg|255=fg].
        """
        img = rgb_image * 1. / 255
        resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        resized = np.expand_dims(resized, axis=0)
        predictions = self.model.predict(resized)
        mask = predictions[0]                       # float 0-1
        mask = (mask > 0.5).astype(np.ubyte) * 255  # threshold 0.5, cast to uchar, rescale [0|255]
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def save_model(self, filename):
        """
        Method to save the whole trained network to disk.

        :param filename: file to save to.
        """
        self.model.save(filename)


def run_rgb_unet_model(logs,
                       data,
                       working,
                       omit,
                       model,
                       save,
                       test,
                       prediction,
                       epochs,
                       batch_size,
                       learning_rate,
                       patience
                       ):
    """
    Helper function to run the RGBUnet model from
    the command line entry point.

    :param logs: directory for log files for tensorboard.
    :param data: root directory of training data.
    :param working: working directory for organising data.
    :param omit: patient identifier to omit, when doing Leave-One-Out.
    :param model: file of previously saved model.
    :param save: file to save model to.
    :param test: input image to test.
    :param prediction: output image, the result of the prediction on test image.
    :param epochs: number of epochs.
    :param batch_size: batch size.
    :param learning_rate: learning rate for optimizer.
    :param patience: number of steps to tolerate non-improving accuracy
    """
    now = datetime.datetime.now()
    date_format = now.today().strftime("%Y-%m-%d")
    time_format = now.time().strftime("%H-%M-%S")
    logfile_name = 'rgbunet-' \
                   + date_format \
                   + '-' \
                   + time_format \
                   + '-' \
                   + str(os.getpid()) \
                   + '.log'

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    file_handler = logging.FileHandler(logfile_name)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    username = getpass.getuser()

    LOGGER.info("Starting RGBUNet version: %s", __version__)
    LOGGER.info("Starting RGBUNet with username: %s.", username)
    LOGGER.info("Starting RGBUNet with platform: %s.", str(platform.uname()))
    LOGGER.info("Starting RGBUNet with cwd: %s.", os.getcwd())
    LOGGER.info("Starting RGBUNet with path: %s.", sys.path)
    LOGGER.info("Starting RGBUNet with save: %s.", save)
    LOGGER.info("Starting RGBUNet with test: %s.", test)
    LOGGER.info("Starting RGBUNet with prediction: %s.", prediction)

    # No point loading network to test an image, if command line args wrong.
    # So, check this up front.
    if test is not None:
        if prediction is None:
            raise ValueError("If you specify a test parameter, you must also "
                             "specify the prediction parameter.")
        if test == prediction:
            raise ValueError("If you specify a test parameter, the value for "
                             "the prediction parameter must be different.")
        if os.path.isfile(prediction) or os.path.isdir(prediction):
            raise ValueError("The prediction parameter should "
                             "be a new file or directory")

    if save is not None:
        dirname = os.path.dirname(save)
        if os.path.exists(dirname) and not os.path.isdir(dirname):
            raise ValueError("Path:" + str(dirname)
                             + " exists, but is not a directory")
        if dirname is not None and len(dirname) > 0 and \
                not os.path.exists(dirname):
            os.makedirs(dirname)

    rgbunet = RGBUNet(logs, data, working, omit, model,
                      learning_rate=learning_rate,
                      epochs=epochs,
                      batch_size=batch_size,
                      patience=patience
                      )

    if save is not None:
        rgbunet.save_model(save)

    if test is not None:

        if os.path.isfile(test):
            test_files = [test]
        elif os.path.isdir(test):
            test_files = ss.get_sorted_files_from_dir(test)
            if not os.path.exists(prediction):
                os.makedirs(prediction)
        else:
            raise ValueError("Invalid value for test parameter ")

        for test_file in test_files:

            img = cv2.imread(test_file)

            start_time = datetime.datetime.now()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = rgbunet.predict(img)

            end_time = datetime.datetime.now()
            time_taken = (end_time - start_time).total_seconds()

            LOGGER.info("Prediction on %s took %s seconds.",
                        test_file, str(time_taken))

            if os.path.isdir(prediction):
                cv2.imwrite(
                    os.path.join(prediction, os.path.basename(test_file)), mask)
            else:
                cv2.imwrite(prediction, mask)
