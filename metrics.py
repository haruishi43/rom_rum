#!/usr/bin/env python3
"""File name: metrics.py

Author: Haruya Ishikawa
Date created: 4/18/2023
Date last modified: 4/18/2023
"""

import math

import numpy as np
from skimage.measure import label


def ROM(gt, pred):
    """Region-wise over-segmentation measure (ROM).

    We count the number of separate contiguous regions for both
    the ground truth and the prediction, and then compute the ratio.

    This metric assumes that the number of classes is 1 where 0 is the background pixel value.

    Args:
        gt (np.ndarray): Ground truth image
        pred (np.ndarray): Segmentation image
    returns:
        float: ROM
    """

    # detect contiguous regions
    gt_regions = label(gt)
    pred_regions = label(pred)

    # number of contiguous regions
    num_gt_regions = len(np.unique(gt_regions)) - 1
    num_pred_regions = len(np.unique(pred_regions)) - 1

    if num_gt_regions == 0 or num_pred_regions == 0:
        return 0

    # count over-segmentation
    num_gt_os = 0
    num_pred_os = 0

    m_o = 0
    for i in range(1, num_gt_regions + 1):
        gt_region = gt_regions == i

        m_count = 0
        for j in range(1, num_pred_regions + 1):
            pred_region = pred_regions == j

            if np.sum(gt_region & pred_region) > 0:
                num_pred_os += 1
                m_count += 1

        if m_count > 1:
            num_gt_os += 1

        m_o += max(0, m_count - 1)

    ror = (num_gt_os / num_gt_regions) * (num_pred_os / num_pred_regions)
    rom = math.tanh(ror * m_o)
    return rom


def RUM(gt, pred):
    """Region-wise under-segmentation measure (RUM).

    We count the number of separate contiguous regions for both
    the ground truth and the prediction, and then compute the ratio.

    This metric assumes that the number of classes is 1 where 0 is the background pixel value.

    Args:
        gt (np.ndarray): Ground truth image
        pred (np.ndarray): Segmentation image
    returns:
        float: RUM
    """

    # detect contiguous regions
    gt_regions = label(gt)
    pred_regions = label(pred)

    # number of contiguous regions
    num_gt_regions = len(np.unique(gt_regions)) - 1
    num_pred_regions = len(np.unique(pred_regions)) - 1

    if num_gt_regions == 0 or num_pred_regions == 0:
        return 0

    # count under-segmentation
    num_gt_us = 0
    num_pred_us = 0

    m_u = 0
    for i in range(1, num_pred_regions + 1):
        pred_region = pred_regions == i

        m_count = 0
        for j in range(1, num_gt_regions + 1):
            gt_region = gt_regions == j

            if np.sum(gt_region & pred_region) > 0:
                num_gt_us += 1
                m_count += 1

        if m_count > 1:
            num_pred_us += 1

        m_u += max(0, m_count - 1)

    rur = (num_gt_us / num_gt_regions) * (num_pred_us / num_pred_regions)
    rum = math.tanh(rur * m_u)
    return rum
