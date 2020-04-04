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
def boxes_centers(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ymin, xmin, ymax, xmax = split_boxes(boxes)
    return (ymax + ymin) / 2, (xmax + xmin) / 2


@jit(nopython=True)
def boxes_heights_widths(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ymin, xmin, ymax, xmax = split_boxes(boxes)
    return (ymax - ymin), (xmax - xmin)


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
