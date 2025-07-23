"""File name: metrics.py

Author: Haruya Ishikawa
Date created: 4/18/2023
Date last modified: 4/18/2023
"""

import math

import numpy as np
from skimage.measure import label


def ROM(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    connectivity: int = 2,
    return_nan: bool = False,
) -> float:
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
    gt_regions = label(gt, connectivity=connectivity)
    pred_regions = label(pred, connectivity=connectivity)

    # number of contiguous regions
    num_gt_regions = np.max(gt_regions)
    num_pred_regions = np.max(pred_regions)

    if num_gt_regions == 0 or num_pred_regions == 0:
        if return_nan:
            return np.nan
        else:
            return 0

    # count over-segmentation
    num_gt_os = 0

    m_o = 0
    oversegmented_pred_indices = set()
    for i in range(1, num_gt_regions + 1):
        gt_region = gt_regions == i

        m_count = 0
        overlapping_preds = []
        for j in range(1, num_pred_regions + 1):
            pred_region = pred_regions == j

            if np.sum(gt_region & pred_region) > 0:
                overlapping_preds.append(j)
                m_count += 1

        if m_count > 1:
            num_gt_os += 1
            for pred_idx in overlapping_preds:
                oversegmented_pred_indices.add(pred_idx)

        m_o += max(0, m_count - 1)

    num_pred_os = len(oversegmented_pred_indices)
    ror = (num_gt_os / num_gt_regions) * (num_pred_os / num_pred_regions)
    rom = math.tanh(ror * m_o)
    return rom


def RUM(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    connectivity: int = 2,
    return_nan: bool = False,
) -> float:
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
    gt_regions = label(gt, connectivity=connectivity)
    pred_regions = label(pred, connectivity=connectivity)

    # number of contiguous regions
    num_gt_regions = np.max(gt_regions)
    num_pred_regions = np.max(pred_regions)

    if num_gt_regions == 0 or num_pred_regions == 0:
        if return_nan:
            return np.nan
        else:
            return 0

    # count under-segmentation
    num_pred_us = 0

    m_u = 0
    undersegmented_gt_indices = set()
    for i in range(1, num_pred_regions + 1):
        pred_region = pred_regions == i

        m_count = 0
        overlapping_gts = []
        for j in range(1, num_gt_regions + 1):
            gt_region = gt_regions == j

            if np.sum(gt_region & pred_region) > 0:
                overlapping_gts.append(j)
                m_count += 1

        if m_count > 1:
            num_pred_us += 1
            for gt_idx in overlapping_gts:
                undersegmented_gt_indices.add(gt_idx)

        m_u += max(0, m_count - 1)

    num_gt_us = len(undersegmented_gt_indices)
    rur = (num_gt_us / num_gt_regions) * (num_pred_us / num_pred_regions)
    rum = math.tanh(rur * m_u)
    return rum
