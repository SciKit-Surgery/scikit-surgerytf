# -*- coding: utf-8 -*-

"""
Module to implement a Fashion MNIST
"""

import logging

LOGGER = logging.getLogger(__name__)


def run_fashion_model(model, test):
    """
    Function to run the Fashion MNIST model,
    as featured on TensorFlow tutorials.
    :param model: file of previously trained weights
    :param test: image to test
    """
    LOGGER.info("Hello model world, model=%s, test=%s", model, test)
