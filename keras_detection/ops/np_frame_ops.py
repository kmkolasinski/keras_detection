from typing import Tuple, Optional
from numba import jit
import numpy as np

# infinitesimally small number
epsilon = 1e-5


@jit(nopython=True)
def split_boxes(
    boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]


@jit(nopython=True)
def split_boxes_nx1(
    boxes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ymin = np.expand_dims(boxes[:, 0], -1)
    xmin = np.expand_dims(boxes[:, 1], -1)
    ymax = np.expand_dims(boxes[:, 2], -1)
    xmax = np.expand_dims(boxes[:, 3], -1)
    return ymin, xmin, ymax, xmax


@jit(nopython=True)
def boxes_centers(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ymin, xmin, ymax, xmax = split_boxes(boxes)
    return (ymax + ymin) / 2, (xmax + xmin) / 2


@jit(nopython=True)
def boxes_heights_widths(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ymin, xmin, ymax, xmax = split_boxes(boxes)
    return (ymax - ymin), (xmax - xmin)


@jit(nopython=True)
def boxes_areas(boxes: np.ndarray) -> np.ndarray:
    ymin, xmin, ymax, xmax = split_boxes(boxes)
    return np.abs(ymax - ymin) * np.abs(xmax - xmin)


@jit(nopython=True)
def clip_coords(coords: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(coords, 0.0), 1.0 - epsilon)


@jit(nopython=True)
def boxes_clipped_centers(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_center, x_center = boxes_centers(boxes)
    # make sure (x, y) are in range [0, 1)
    y_center = clip_coords(y_center)
    x_center = clip_coords(x_center)
    return y_center, x_center


@jit(nopython=False)
def boxes_scale(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scale = np.array([[sy, sx, sy, sx]], dtype=boxes.dtype)
    return boxes * scale


@jit(nopython=True)
def is_box_in_window(
    boxes: np.ndarray, window: Optional[np.ndarray] = None
) -> np.ndarray:
    yc, xc = boxes_centers(boxes)
    if window is None:
        window = np.array([0.0, 0.0, 1.0, 1.0], dtype=boxes.dtype)

    w_ymin, w_xmin, w_ymax, w_xmax = window
    x_in_window = np.logical_and(xc >= w_xmin, xc <= w_xmax)
    y_in_window = np.logical_and(yc >= w_ymin, yc <= w_ymax)

    return x_in_window * y_in_window


@jit(nopython=True)
def intersection_area(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute intersection areas between boxes of two boxes list.

    Args:
        boxes_a: N boxes
        boxes_b: M boxes

    Returns:
      a tensor with shape [N, M] representing areas of all intersections
    """
    y_min1, x_min1, y_max1, x_max1 = split_boxes_nx1(boxes_a)
    y_min2, x_min2, y_max2, x_max2 = split_boxes_nx1(boxes_b)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@jit(nopython=True)
def iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes intersection-over-union between boxes of two collections.

    Args:
        boxes_a: N boxes
        boxes_b: M boxes

    Returns:
        a matrix with shape [N, M] representing iou scores.
    """
    intersections = intersection_area(boxes_a, boxes_b)
    areas_one = boxes_areas(boxes_a)
    areas_two = boxes_areas(boxes_b)
    unions = np.expand_dims(areas_one, 1) + np.expand_dims(areas_two, 0) - intersections
    return intersections / unions


def argmax_iou_matching(
    boxes_a: np.ndarray, boxes_b: np.ndarray, iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        boxes_a: N boxes
        boxes_b: M boxes
        iou_threshold:

    Returns:
        a_matches: match array of length N, -1 is set to not matched
        b_matches: match array of length M, -1 is set to not matched
    """

    if boxes_a.shape[0] == 0:
        return np.zeros([0], dtype=np.int64), -np.ones([boxes_b.shape[0]], dtype=np.int64)
    if boxes_b.shape[0] == 0:
        return -np.ones([boxes_a.shape[0]], dtype=np.int64), np.zeros([0], dtype=np.int64)

    a_matches = np.zeros([boxes_a.shape[0]], dtype=np.int64)
    b_matches = np.zeros([boxes_b.shape[0]], dtype=np.int64)

    iou_matrix = iou(boxes_a, boxes_b)
    num_a_boxes = boxes_a.shape[0]

    a_matches[:] = -1
    b_matches[:] = -1

    a_best_matches = np.argmax(iou_matrix, 1)
    b_best_matches = np.argmax(iou_matrix, 0)

    for i in range(num_a_boxes):
        best_match = a_best_matches[i]
        if (
            iou_matrix[i, best_match] > iou_threshold
            and b_best_matches[best_match] == i
        ):
            a_matches[i] = best_match
            b_matches[best_match] = i

    return a_matches, b_matches
