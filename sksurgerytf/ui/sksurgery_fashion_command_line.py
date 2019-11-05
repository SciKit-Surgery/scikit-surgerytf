# coding=utf-8

""" Command line entry point for sksurgeryfashion demo. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.models.fashion as f


def main(args=None):
    """
    Entry point for sksurgeryfashion demo.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgeryfashion')

    parser.add_argument("-l", "--logs",
                        required=False,
                        type=str,
                        default="logs/fit/",
                        help="Log directory for tensorboard.")

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
                        help="Test image (28 x 28), single channel.")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryfashion version ' + friendly_version_string)

    args = parser.parse_args(args)

    f.run_fashion_model(args.logs,
                        args.model,
                        args.save,
                        args.test
                        )
