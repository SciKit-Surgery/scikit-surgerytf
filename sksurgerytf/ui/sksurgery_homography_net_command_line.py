# coding=utf-8

""" Command line entry point for sksurgeryhomographynet demo. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.models.homography_net as m


def main(args=None):
    """
    Entry point for sksurgeryhomographynet demo.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgeryhomographynet')

    parser.add_argument("-l", "--logs",
                        required=False,
                        type=str,
                        default="logs/fit/",
                        help="Log directory for tensorboard.")

    parser.add_argument("-w", "--weights",
                        required=False,
                        type=str,
                        help="Load pre-trained weights file.")

    parser.add_argument("-a", "--imageA",
                        required=False,
                        type=str,
                        help="Test image A, RGB.")

    parser.add_argument("-b", "--imageB",
                        required=False,
                        type=str,
                        help="Test image B, RGB.")

    parser.add_argument("-s", "--save",
                        required=False,
                        type=str,
                        help="Save weights file.")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgeryhomographynet version ' + friendly_version_string)

    args = parser.parse_args(args)

    m.run_homography_net_model(args.logs,
                               args.weights,
                               args.imageA,
                               args.imageB,
                               args.save)
