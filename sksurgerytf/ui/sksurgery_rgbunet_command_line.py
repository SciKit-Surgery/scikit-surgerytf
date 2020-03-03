# coding=utf-8

""" Command line entry point for 2D RGB Unet script. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.models.rgb_unet as unet


def main(args=None):
    """
    Entry point for sksurgeryrgbunet script.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgeryliverseg')

    parser.add_argument("-l", "--logs",
                        required=False,
                        type=str,
                        default="logs/fit/",
                        help="Log directory for tensorboard.")

    parser.add_argument("-d", "--data",
                        required=False,
                        type=str,
                        help="Root directory of data to train on.")

    parser.add_argument("-w", "--working",
                        required=False,
                        type=str,
                        default="logs/working",
                        help="Root directory to write intermediate output to.")

    parser.add_argument("-o", "--omit",
                        required=False,
                        type=str,
                        help="Directory identifier to omit for Leave-One-Out.")

    parser.add_argument("-m", "--model",
                        required=False,
                        type=str,
                        help="Load pre-trained model (normally .hdf5).")

    parser.add_argument("-s", "--save",
                        required=False,
                        type=str,
                        help="Save trained model (normally .hdf5).")

    parser.add_argument("-t", "--test",
                        required=False,
                        type=str,
                        help="Test/predict input image, RGB, .png, .jpg")

    parser.add_argument("-p", "--prediction",
                        required=False,
                        type=str,
                        help="Test/predict output image, RGB, .png, .jpg")

    parser.add_argument("-e", "--epochs",
                        required=False,
                        type=int,
                        default=50,
                        help="Number of epochs.")

    parser.add_argument("-b", "--batchsize",
                        required=False,
                        type=int,
                        default=2,
                        help="Batch size.")

    parser.add_argument("-r", "--learningrate",
                        required=False,
                        type=float,
                        default=0.0001,
                        help="Learning rate for optimizer (Adam).")

    parser.add_argument("-pat", "--patience",
                        required=False,
                        type=int,
                        default=5,
                        help="Patience (early stopping tolerance, #steps.)")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryrgbunet version ' + friendly_version_string)

    args = parser.parse_args(args)

    unet.run_rgb_unet_model(args.logs,
                            args.data,
                            args.working,
                            args.omit,
                            args.model,
                            args.save,
                            args.test,
                            args.prediction,
                            args.epochs,
                            args.batchsize,
                            args.learningrate,
                            args.patience
                            )
