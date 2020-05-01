from typing import Tuple
from numba import jit
import numpy as np

BLEND_MULTIPLY = 1
BLEND_ADD = 2
BLEND_REPLACE = 3


@jit("float32[:](float32[:])", nopython=True)
def clip_color(color: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(color, 0.0), 1.0).astype(np.float32)


@jit(nopython=True)
def image_height_width(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[:2]


@jit(nopython=True)
def clip_int(value: int, max_val: int) -> np.ndarray:
    return np.minimum(np.maximum(value, 0), max_val)


@jit(nopython=True)
def clip_rect(rect: np.ndarray, height: int, width: int) -> np.ndarray:
    ymin, xmin, ymax, xmax = rect
    ymin, ymax = clip_int(ymin, height), clip_int(ymax, height)
    xmin, xmax = clip_int(xmin, width), clip_int(xmax, width)
    return np.array([ymin, xmin, ymax, xmax])


@jit(nopython=True)
def clip_bbox(coords: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(coords, 0.0), 1.0)


@jit(nopython=True)
def clip_rect_to_image(rect: np.ndarray, image: np.ndarray) -> np.ndarray:
    h, w = image_height_width(image)
    return clip_rect(rect, h - 1, w - 1)


@jit(nopython=True)
def sample_min_max(min_max: Tuple[float, float]) -> float:
    dh = min_max[1] - min_max[0]
    return np.random.rand() * dh + min_max[0]


@jit(nopython=True)
def draw_rectangle(
    image: np.ndarray,
    rect: np.ndarray,
    color: np.ndarray,
    alpha: float = 0.5,
    mode: int = BLEND_ADD,
) -> np.ndarray:

    ymin, xmin, ymax, xmax = rect
    if mode == BLEND_ADD:
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                curr_color = image[i, j, :]
                new_color = clip_color(curr_color + color)
                image[i, j, :] = curr_color * alpha + (1 - alpha) * new_color
    elif mode == BLEND_MULTIPLY:
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                curr_color = image[i, j, :]
                new_color = curr_color * color
                image[i, j, :] = curr_color * alpha + (1 - alpha) * new_color
    else:
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                curr_color = image[i, j, :]
                new_color = color
                image[i, j, :] = curr_color * alpha + (1 - alpha) * new_color

    return image


@jit(nopython=True)
def draw_t_shape(
    image: np.ndarray,
    rect: np.ndarray,
    color: np.ndarray,
    shape_type: int,
    alpha: float = 0.5,
    mode: int = BLEND_ADD,
) -> np.ndarray:

    row, col = divmod(shape_type, 3)
    ymin, xmin, ymax, xmax = rect
    height, width = (ymax - ymin) / 3, (xmax - xmin) / 3

    hrect = ymin + height * row, xmin, ymin + (row + 1) * height, xmax
    vrect = ymin, xmin + width * col, ymax, xmin + (col + 1) * width

    hrect = clip_rect_to_image(np.array(hrect, dtype=np.int64), image)
    vrect = clip_rect_to_image(np.array(vrect, dtype=np.int64), image)
    for rect in [hrect, vrect]:
        image = draw_rectangle(
            image=image, rect=rect, color=color, alpha=alpha, mode=mode
        )
    return image


@jit(nopython=True)
def coord_outside_window(coord: float) -> float:
    dcoord = 0.0
    if coord < 0:
        dcoord = abs(coord)
    elif coord > 1:
        dcoord = 1.0 - abs(coord)
    return dcoord


@jit(nopython=True)
def draw_random_t_shape(
    image: np.ndarray,
    color: np.ndarray,
    min_max_ypos: Tuple[float, float] = (0.0, 1.0),
    min_max_xpos: Tuple[float, float] = (0.0, 1.0),
    min_max_height: Tuple[float, float] = (0.1, 0.2),
    min_max_hw_ratio: Tuple[float, float] = (0.8, 1.2),
    alpha: float = 0.5,
    mode: int = BLEND_ADD,
) -> Tuple[np.ndarray, np.ndarray, int]:

    shape_type = np.random.randint(0, 9)
    height = sample_min_max(min_max_height)
    ratio = sample_min_max(min_max_hw_ratio)
    width = height / ratio

    ymin = sample_min_max(min_max_ypos) - height / 2
    xmin = sample_min_max(min_max_xpos) - width / 2
    ymax = ymin + height
    xmax = xmin + width

    dymin = coord_outside_window(ymin)
    dymax = coord_outside_window(ymax)
    dxmin = coord_outside_window(xmin)
    dxmax = coord_outside_window(xmax)

    im_h, im_w = image_height_width(image)
    dy, dx = dymin + dymax, dxmin + dxmax
    bbox = np.array([ymin + dy, xmin + dx, ymax + dy, xmax + dx])
    image_rect = bbox * np.array([im_h, im_w, im_h, im_w])

    image = draw_t_shape(
        image=image,
        rect=image_rect.astype(np.int64),
        color=color,
        shape_type=shape_type,
        alpha=alpha,
        mode=mode,
    )
    return image, clip_bbox(bbox), shape_type


@jit(nopython=True)
def draw_random_t_shape_image(
    image: np.ndarray,
    boxes_output: np.ndarray,
    labels_output: np.ndarray,
    min_max_height: Tuple[float, float] = (0.1, 0.2),
    min_max_hw_ratio: Tuple[float, float] = (0.8, 1.2),
    alpha: float = 0.5,
    mode: int = BLEND_ADD,
    num_colors: int = 32,
    color_min_value: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_boxes = boxes_output.shape[0]
    grid_size = round(num_boxes ** 0.5)

    color_min_value = color_min_value // 255
    dg = 0.5 / grid_size

    for i in range(num_boxes):
        row, col = divmod(i, grid_size)
        row, col = row / grid_size, col / grid_size

        min_max_ypos = (max(row, 0.0), min(row + 2 * dg, 1.0))
        min_max_xpos = (max(col, 0.0), min(col + 2 * dg, 1.0))

        n = 1 / (num_colors - 1)
        r = n * np.random.randint(color_min_value, num_colors)
        g = n * np.random.randint(color_min_value, num_colors)
        b = n * np.random.randint(color_min_value, num_colors)
        color = np.array([r, g, b], dtype=np.float32)

        image, bbox, label = draw_random_t_shape(
            image=image,
            color=color,
            min_max_ypos=min_max_ypos,
            min_max_xpos=min_max_xpos,
            min_max_height=min_max_height,
            min_max_hw_ratio=min_max_hw_ratio,
            alpha=alpha,
            mode=mode,
        )
        boxes_output[i] = bbox
        labels_output[i] = label
    return image, boxes_output, labels_output


def create_random_rectangles_dataset_generator(
    image_size: Tuple[int, int] = (224, 224),
    min_max_num_boxes: Tuple[int, int] = (5, 30),
    min_max_height: Tuple[float, float] = (0.08, 0.15),
    min_max_hw_ratio: Tuple[float, float] = (0.8, 1.2),
    alpha: float = 0.5,
    mode: int = BLEND_MULTIPLY,
    num_colors: int = 32,
    color_min_value: int = 50
):

    assert color_min_value // 255 < num_colors
    assert min_max_num_boxes[0] >= 1

    while True:

        image = np.ones([image_size[0], image_size[1], 3])
        bg_color = np.random.randint(1 + color_min_value // 255, num_colors, [1, 1, 3])
        bg_color = bg_color / bg_color.max()
        image = image * bg_color

        num_boxes = np.random.randint(*min_max_num_boxes)
        boxes = np.zeros([num_boxes, 4], dtype=np.float32)
        labels = np.zeros([num_boxes], dtype=np.int64)

        image, boxes, labels = draw_random_t_shape_image(
            image.astype(np.float32),
            boxes.astype(np.float32),
            labels,
            min_max_height=min_max_height,
            min_max_hw_ratio=min_max_hw_ratio,
            alpha=alpha,
            mode=mode,
            num_colors=num_colors,
            color_min_value=color_min_value
        )

        weights = np.ones_like(labels).astype(np.float32)
        image = np.array(255 * image, dtype=np.uint8)
        features = {"image": image}
        labels = {"boxes": boxes, "labels": labels, "weights": weights}
        yield {"features": features, "labels": labels}
