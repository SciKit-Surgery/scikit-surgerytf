# -*- coding: utf-8 -*-

"""
Module to implement a basic classifier for the Fashion MNIST dataset.
The aim of this module is to demonstrate how to create a class
that can be developed, tested and re-used effectively. It is not
a demonstration on how to do deep learning, or classification per se.

Inspired by
`TensorFlow tutorials
<https://www.tensorflow.org/tutorials/keras/classification>`_.

"""
import os
import sys
import logging
import copy
import datetime
import platform
import ssl
from pathlib import Path
import numpy as np
from tensorflow import keras
import cv2


from sksurgerytf import __version__

LOGGER = logging.getLogger(__name__)


class FashionMNIST:
    """
    Class to encapsulate a classifier for the Fashion MNIST dataset.
    """
    def __init__(self,
                 logs="logs/fit",
                 weights=None,
                 learning_rate=0.001,
                 epochs=1
                 ):
        """
        Class to implement a 10-class classifier for the Fashion MNIST dataset,
        and provide entry points for both training and testing.

        If the constructor is called without weights, the data is loaded
        and a full training cycle is performed in order to learn the weights.

        If the constructor is called with weights, these are loaded, as is,
        with no further training.

        :param logs: relative path to folder to write tensorboard log files.
        :param weights: file name prefix of pre-saved weights.
        :param learning_rate: float, default=0.001 which is Keras default.
        :param epochs: int, default=1
        """
        self.logs = logs
        self.learning_rate = learning_rate
        self.epochs = epochs

        LOGGER.info("Creating FashionMNIST with log dir: %s.",
                    str(self.logs))
        LOGGER.info("Creating FashionMNIST with weights file: %s.",
                    str(weights))
        LOGGER.info("Creating FashionMNIST with learning_rate: %s.",
                    str(self.learning_rate))
        LOGGER.info("Creating FashionMNIST with epochs: %s.",
                    str(self.epochs))

        # To fix issues with SSL certificates
        ssl._create_default_https_context = ssl._create_unverified_context

        self.model = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover',
                            'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                            'Bag', 'Ankle boot']

        self.build_model()

        if weights is not None:
            self.model.load_weights(weights)
        else:
            self.load_data()
            self.train()

    def get_class_names(self):
        """
        Returns a copy of the valid class names. We return copies
        to stop external people accidentally editing the internal copies.
        It's safer in the long run, even if its easy to work around.

        :return: list of strings
        """
        return copy.deepcopy(self.class_names)

    def load_data(self):
        """
        Loads the data.

        fashion_mnist is available from TF/Keras directly, and fortunately,
        it get's cached on your computer. So, subsequent calls are fast.
        """
        (self.train_images, self.train_labels), \
            (self.test_images, self.test_labels) = \
            keras.datasets.fashion_mnist.load_data()

        self.train_images = self.__preprocess_data(self.train_images)
        self.test_images = self.__preprocess_data(self.test_images)

    @staticmethod
    def __preprocess_data(images):
        """
        Pre-processes the data. For this data set we just:

          - Normalise unsigned char [0-255] to float [0-1].

        :param images: (m x 28 x 28) numpy, single channel, [0-255], uchar
        :return: normalised (m x 28 x 28) numpy, single channel, [0-1], float
        """
        return images / 255.0

    def build_model(self):
        """
        Constructs the neural network.

          - 128 node FC + relu
          - 10 node FC + softmax

        For the purpose of this demo, the network is irrelevant.
        Its the same one, copied from
        `TensorFlow tutorials
        <https://www.tensorflow.org/tutorials/keras/classification>`_.

        Other examples you could experiment with include
        `this one
        <https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a>`_.
        """
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        optimiser = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=optimiser,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):
        """
        Method to train the neural network.

        Uses:

          - Adam optimiser
          - sparse_categorical_crossentropy cost function
          - evaluates accuracy metric

        :return: output of self.model.evaluate on test set.
        """

        log_dir = os.path.join(Path(self.logs),
                               datetime.datetime.now()
                               .strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=1)

        self.model.fit(self.train_images,
                       self.train_labels,
                       epochs=self.epochs,
                       validation_data=(self.test_images, self.test_labels),
                       callbacks=[tensorboard_callback]
                       )

        return self.model.evaluate(self.test_images,
                                   self.test_labels,
                                   verbose=2,
                                   )

    def test(self, image):
        """
        Method to test a single (28 x 28) image.

        :param image: (28 x 28), numpy, single channel, [0-255], uchar.
        :return: (class_index, class_name)
        """
        normalised = self.__preprocess_data(image)
        img = (np.expand_dims(normalised, 0))
        predictions = self.model.predict(img)
        class_index = np.argmax(predictions[0])
        return class_index, self.class_names[class_index]

    def save_weights(self, filename):
        """
        Method to save the network weights to disk.

        :param filename: file to save to
        """
        self.model.save_weights(filename)

    def get_test_image(self, index):
        """
        Extracts an image from the test data. Useful for unit testing,
        as the original data comes packaged up in a zip file.

        :param index: int [0-9999], unchecked
        :return: image, (28 x 28), numpy, single channel, [0-255], uchar.
        """
        if self.test_images is None:
            self.load_data()
        img = self.test_images[index, :, :]
        reshaped = img.reshape([28, 28])
        rescaled = reshaped * 255
        output = rescaled.astype(np.uint8)
        return output

    def extract_failures(self, number_to_fetch):
        """
        Returns incorrectly classified test images.
        :param number_to_fetch: int, the number to find.

        This method is slow, its only for demo purposes.

        :return: indexes, images, predicted, labels
        """
        indexes = []
        images = []
        predicted = []
        labels = []

        if self.test_images is None:
            self.load_data()
        for counter in range(self.test_images.shape[0]):
            image = self.get_test_image(counter)
            class_index, _ = self.test(image)
            if class_index != self.test_labels[counter]:
                indexes.append(counter)
                images.append(image)
                predicted.append(class_index)
                labels.append(self.test_labels[counter])
                if len(indexes) == number_to_fetch:
                    break
        return indexes, images, predicted, labels


def run_fashion_model(logs,
                      weights,
                      image,
                      save):
    """
    Helper function to run the Fashion MNIST model from
    the command line entry point.

    :param logs: directory for log files for tensorboard.
    :param weights: file of previously trained weights
    :param image: image to test
    :param save: file to save weights to
    """
    # pylint: disable=line-too-long
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    LOGGER.info("Starting fashion.py version: %s", __version__)
    LOGGER.info("Starting fashion.py with platform: %s.", str(platform.uname()))
    LOGGER.info("Starting fashion.py with cwd: %s.", os.getcwd())
    LOGGER.info("Starting fashion.py with path: %s.", sys.path)

    fmn = FashionMNIST(logs, weights)

    if save is not None:
        fmn.save_weights(save)

    if image is not None:
        img = cv2.imread(image)
        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        class_index, class_name = fmn.test(greyscale)
        LOGGER.info("Image: %s, categorised as: %s:%s",
                    image, class_index, class_name)
