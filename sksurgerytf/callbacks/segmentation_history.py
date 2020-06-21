# -*- coding: utf-8 -*-

"""
Module to implement callback to save an image, with segmentation.
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf


class SegmentationHistory(keras.callbacks.Callback):
    """
    Class to implement Tensorboard callback to save a batch of images and their
    segmentations, so we can monitor progress directly in Tensorboard.
    """
    def __init__(self,
                 tensor_board_dir,
                 data,
                 number_of_samples,
                 desired_number_images):
        """
        Constructor.
        :param tensor_board_dir: directory to log to
        :param data: an ImageDataGenerator
        :param number_of_samples: number of samples coming from generator.
        :param desired_number_images: the number of images you want logging.
        """
        super(SegmentationHistory, self).__init__()
        if number_of_samples <= 0:
            raise ValueError('number_of_samples must be > 0')
        if desired_number_images < 1:
            raise ValueError('desired_number_images must be >= 1')

        self.tensor_board_dir = tensor_board_dir
        self.data = data
        self.number_of_samples = number_of_samples
        self.desired_number_images = desired_number_images
        self.modulo = number_of_samples // desired_number_images

    # pylint: disable=unused-argument
    #pylint:disable=signature-differs
    def on_epoch_end(self, epoch, logs):
        """
        Called at the end of each epoch, so we can log data.
        :param epoch: number of the epoch
        :param logs: logging info, see docs, currently unused.
        """
        images = []
        labels = []
        counter = 0
        for item in self.data:
            image_data = item[0]
            label_data = item[1]
            if counter % self.modulo == 0 \
                    and len(images) < self.desired_number_images:
                pred = self.model.predict(image_data)
                mask = pred[0]
                mask = (mask > 0.5).astype(np.ubyte) * 255
                images.append(mask)
                labels.append(label_data[0])
            counter = counter + 1
            if counter >= self.number_of_samples:
                break
        images_concatenated = np.concatenate(images, axis=1)
        labels_concatenated = np.concatenate(labels, axis=1)
        data = np.concatenate((images_concatenated,
                               labels_concatenated), axis=0)
        self.save_to_tensorboard(data, epoch)

    def save_to_tensorboard(self, npyfile, step):
        """
        Write a set of images, in a format suitable for Tensorboard.

        :param npyfile: block of data, see above method.
        :param step: some int to indicate progress, e.g. batch number or epoch.
        """
        #pylint:disable=not-context-manager
        image = np.reshape(npyfile, (-1,
                                     npyfile.shape[0],
                                     npyfile.shape[1],
                                     npyfile.shape[2]))
        writer = tf.summary.create_file_writer(self.tensor_board_dir)
        with writer.as_default():
            tf.summary.image("Predicted (top), Labelled (bottom)",
                             image, step=step)
