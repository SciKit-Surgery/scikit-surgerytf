# coding=utf-8

""" Command line entry point for sksurgeryliverseg script. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.models.liverseg as ls


def main(args=None):
    """
    Entry point for sksurgeryliverseg demo.

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
                        help="Root directory of data to train on.")

    parser.add_argument("-o", "--omit",
                        required=False,
                        type=str,
                        help="Directory identifier to omit.")

    parser.add_argument("-m", "--model",
                        required=False,
                        type=str,
                        help="Load complete pre-trained model (normally .hd5).")

    parser.add_argument("-s", "--save",
                        required=False,
                        type=str,
                        help="Save model (normally .hd5).")

    parser.add_argument("-t", "--test",
                        required=False,
                        type=str,
                        help="Test image (1920 x 540), RGB.")

    parser.add_argument("-e", "--epochs",
                        required=False,
                        type=int,
                        default=10,
                        help="Number of epochs")

    parser.add_argument("-b", "--batchsize",
                        required=False,
                        type=int,
                        default=4,
                        help="Batch size")

    parser.add_argument("-r", "--learningrate",
                        required=False,
                        type=float,
                        default=0.0001,
                        help="Learning rate for optimizer (Adam).")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryliverseg version ' + friendly_version_string)

    args = parser.parse_args(args)

    ls.run_liverseg_model(args.logs,
                          args.data,
                          args.working,
                          args.omit,
                          args.model,
                          args.save,
                          args.test,
                          args.epochs,
                          args.batchsize,
                          args.learningrate
                          )
