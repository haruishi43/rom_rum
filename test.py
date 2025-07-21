import numpy as np
import pytest
import math

from metrics import RUM

# Test cases for RUM
# Each tuple contains: (test_id, gt_array, pred_array, expected_rum_value)
RUM_TEST_CASES = [
    (
        "perfect_match",
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
        0.0,
    ),
    (
        "no_foreground_gt",
        np.zeros((5, 5), dtype=int),
        np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        0.0,
    ),
    (
        "no_foreground_pred",
        np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        np.zeros((5, 5), dtype=int),
        0.0,
    ),
    (
        "no_foreground_both",
        np.zeros((5, 5), dtype=int),
        np.zeros((5, 5), dtype=int),
        0.0,
    ),
    (
        "simple_under_segmentation_2_to_1",
        np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ]),
        np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]),
        # num_gt=2, num_pred=1. pred_1 merges gt_1, gt_2.
        # m_count=2, m_u=1. num_pred_us=1. num_gt_us=2.
        # rur = (2/2)*(1/1) = 1. rum = tanh(1*1) = tanh(1)
        math.tanh(1.0),
    ),
    (
        "severe_under_segmentation_4_to_1",
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1]
        ]),
        np.ones((5, 5), dtype=int),
        # num_gt=4, num_pred=1. pred_1 merges 4 gt regions.
        # m_count=4, m_u=3. num_pred_us=1. num_gt_us=4.
        # rur = (4/4)*(1/1) = 1. rum = tanh(1*3) = tanh(3)
        math.tanh(3.0),
    ),
    (
        "no_under_segmentation_is_over_segmentation",
        np.ones((5, 5), dtype=int),
        np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1]
        ]),
        # This is an over-segmentation case, so RUM should be 0.
        0.0,
    ),
    (
        "mixed_segmentation_3_to_2",
        np.array([
            [1, 1, 0, 0, 0],  # GT 1
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],  # GT 2
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]  # GT 3
        ]),
        np.array([
            [1, 1, 1, 1, 0],  # Pred 1 (merges GT 1 and 2)
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1]  # Pred 2 (matches GT 3)
        ]),
        # num_gt=3, num_pred=2.
        # Pred 1 merges GT 1,2. m_count=2, m_u=1. num_pred_us=1. num_gt_us=2.
        # Pred 2 matches GT 3. m_count=1, m_u=0.
        # Total m_u=1.
        # rur = (2/3)*(1/2) = 1/3. rum = tanh((1/3)*1) = tanh(1/3)
        math.tanh(1.0 / 3.0),
    ),
    (
        "disjoint_no_overlap",
        np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
        0.0,
    ),
]


@pytest.mark.parametrize("test_id, gt, pred, expected_rum", RUM_TEST_CASES, ids=[case[0] for case in RUM_TEST_CASES])
def test_RUM(test_id, gt, pred, expected_rum):
    """
    Tests the Region-wise Under-segmentation Measure (RUM) with various scenarios.
    """
    # Act
    actual_rum = RUM(gt, pred)

    # Assert
    assert actual_rum == pytest.approx(expected_rum)
