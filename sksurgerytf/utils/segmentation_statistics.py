# -*- coding: utf-8 -*-

"""
Module to implement various segmentation statistics for evaluation.
"""
import os
import numpy as np
import cv2


def check_same_size(image_a, image_b):
    """
    Check shape the same.

    :param image_a:  image
    :param image_b: image
    :return:
    """
    image_a_shape = image_a.shape
    image_b_shape = image_b.shape
    if image_a_shape != image_b_shape:
        raise ValueError('image_a shape:' + str(image_a_shape)
                         + ' != image_b shape:' + str(image_b_shape))


def get_sorted_files_from_dir(directory):
    """
    Retrieves all files in directory, sorted.
    :param directory: directory path name
    :return: list of file names
    """
    names = os.listdir(directory)
    names.sort()
    result = []
    for name in names:
        result.append(os.path.join(directory, name))
    return result


def get_confusion_matrix(gold_standard, segmentation):
    """
    Compute the confusion matrix of 2 boolean images.

    Inspired by NiftyNet.

    :param gold_standard: gold standard / reference image.
    :param segmentation: segmented / predicted / inferred image.
    :return: 2x2 confusion matrix, [[TN, FN],[FP,TP]].
    """
    lnot = np.logical_not
    land = np.logical_and
    confusion = np.array(
        [[np.sum(land(lnot(gold_standard), lnot(segmentation))),  # TN
          np.sum(land(lnot(gold_standard), segmentation))],       # FN
         [np.sum(land(gold_standard, lnot(segmentation))),        # FP
          np.sum(land(gold_standard, segmentation))]])            # TP
    return confusion


def calculate_dice(gold_standard, segmentation):
    """
    Computes dice score of two boolean images.

    Inspired by NiftyNet.

    :param gold_standard: gold standard / reference image.
    :param segmentation: segmented / predicted / inferred image.
    :return: dice score
    """
    mat = get_confusion_matrix(gold_standard, segmentation)
    return 2 * mat[1, 1] / (mat[1, 1] * 2 + mat[1, 0] + mat[0, 1])


def run_seg_stats(gold_standard, segmentation):
    """
    Compares segmentation image(s) to gold standard images(s).

    :param gold_standard: directory, or single image
    :param segmentation: directory, or single image
    :return: list of stats
    """
    if os.path.isfile(gold_standard) and os.path.isfile(segmentation):
        gold_standard_files = [gold_standard]
        segmentation_files = [segmentation]
    elif os.path.isdir(gold_standard) and os.path.isdir(segmentation):
        gold_standard_files = get_sorted_files_from_dir(gold_standard)
        segmentation_files = get_sorted_files_from_dir(segmentation)
    else:
        raise ValueError("Arguments should be both "
                         "file names or both directories")

    if len(gold_standard_files) != len(segmentation_files):
        raise ValueError("Lists of unequal length, so can't compare")

    # At the moment, only checking dice. More to follow!
    width = 1
    results = np.zeros((0, width))

    number_of_files = len(gold_standard_files)

    for counter in range(number_of_files):
        gold_standard_file = gold_standard_files[counter]
        gold_standard_image = cv2.imread(gold_standard_file)
        gold_standard_image = cv2.cvtColor(gold_standard_image,
                                           cv2.COLOR_BGR2GRAY)
        segmented_file = segmentation_files[counter]
        segmented_image = cv2.imread(segmented_file)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        check_same_size(gold_standard_image, segmented_image)
        dice = calculate_dice(gold_standard_image > 0,
                              segmented_image > 0)
        if not np.isnan(dice):
            new_row = np.zeros((1, width))
            new_row[0][0] = dice
            results = np.vstack([results, new_row])
        else:
            print("Image " + gold_standard_files[counter]
                  + " is possibly blank")

    if results.shape[0] > 1:
        if results.shape[0] != number_of_files:
            print("Warning, number_of_files=" + str(number_of_files)
                  + ", but averaging over " + str(results.shape[0])
                  + ", so check for completely blank masks?"
                  )
        summaries = np.zeros((3, width))
        summaries[0] = np.average(results, axis=0)
        summaries[1] = np.std(results, axis=0)
        summaries[2] = np.median(results, axis=0)
        return summaries.T

    return results
