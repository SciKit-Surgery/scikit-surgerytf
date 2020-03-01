# coding=utf-8

""" Command line entry point for sksurgerysegstats script. """

import argparse
from sksurgerytf import __version__
import sksurgerytf.utils.segmentation_statistics as ss


def main(args=None):
    """
    Entry point for sksurgerysegstats script.

    Keep as little code as possible in this file, as it's hard to unit test.
    """
    parser = argparse.ArgumentParser(description='sksurgerysegstats')

    parser.add_argument("-g", "--gold",
                        required=True,
                        type=str,
                        help="Image or directory of images for gold standard.")

    parser.add_argument("-s", "--segmented",
                        required=True,
                        type=str,
                        help="Image or directory of segmented images.")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "--version",
        action='version',
        version='sksurgerysegstats version ' + friendly_version_string)

    args = parser.parse_args(args)

    results = ss.run_seg_stats(args.gold,
                               args.segmented
                               )

    print(results)
