# -*- coding: utf-8 -*-

import os
from tensorflow import keras
import pytest
import cv2
import numpy as np


def test_shuffle():

    train_data_gen_args = dict(horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='nearest',
                               rotation_range=10,
                               width_shift_range=0.05,
                               height_shift_range=0.05,
                               zoom_range=[0.9, 1.0])

    train_image_datagen = keras.preprocessing.image.ImageDataGenerator(
        **train_data_gen_args)
    train_mask_datagen = keras.preprocessing.image.ImageDataGenerator(
        **train_data_gen_args)

    seed = 1
    shuffle = True

    train_image_generator = train_image_datagen.flow_from_directory(
        os.path.dirname('tests/data/segmentation/images/object'),
        target_size=(512, 512),
        batch_size=1,
        color_mode='rgb',
        class_mode=None,
        shuffle=shuffle,
        seed=seed)

    train_mask_generator = train_mask_datagen.flow_from_directory(
        os.path.dirname('tests/data/segmentation/masks/object'),
        target_size=(512, 512),
        batch_size=1,
        color_mode='grayscale',
        class_mode=None,
        shuffle=shuffle,
        seed=seed)

    train_generator = zip(train_image_generator, train_mask_generator)

    for x, y in train_generator:
        image = x[0].astype(np.ubyte)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = y[0].astype(np.ubyte)
        cv2.imwrite(os.path.join('tests/output/', 'image.png'), bgr_image)
        cv2.imwrite(os.path.join('tests/output/', 'mask.png'), mask)
        print("Checking shuffle)")
