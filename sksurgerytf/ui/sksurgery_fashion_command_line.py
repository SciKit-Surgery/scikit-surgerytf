# coding=utf-8

""" Command line processing for sksurgeryfashion. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.models.fashion as f


def main(args=None):

    """
    Entry point for sksurgeryfashion command line application.
    """

    parser = argparse.ArgumentParser(description='sksurgeryfashion')

    parser.add_argument("-m", "--model",
                        required=True,
                        type=str,
                        help="Model file.")

    parser.add_argument("-t", "--test",
                        required=False,
                        type=str,
                        help="Test a specific image.")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryfashion version ' + friendly_version_string)

    args = parser.parse_args(args)

    f.run_fashion_model(args.m, args.t)
