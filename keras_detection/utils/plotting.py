import colorsys
import io
import itertools
from typing import Optional, Tuple, Union, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import patches

from keras_detection.ops.np_frame_ops import boxes_heights_widths


def plot_to_image(figure, format: str = "png"):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""

    buf = io.BytesIO()
    plt.savefig(buf, format=format)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return image


def resize_nearest_neighbour(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, resample=Image.NEAREST))


def draw_objectness_map(
    image: Union[np.ndarray, tf.Tensor],
    objectness: Union[np.ndarray, tf.Tensor],
    figsize: Tuple[int, int] = (5, 5),
    fmt: str = "png",
    subtitle: str = "",
    **kwargs,
) -> Image.Image:

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(objectness, tf.Tensor):
        objectness = objectness.numpy()

    height, width = image.shape[:2]

    if objectness.shape[-1] == 1 and len(objectness.shape) == 3:
        objectness = objectness[..., 0]

    figure = plt.figure(figsize=figsize)
    plt.title(f"Objectness map {objectness.shape} {subtitle}")
    plt.imshow(image)

    objectness = (np.clip(objectness, 0, 1) * 255).astype("uint8")
    objectness = resize_nearest_neighbour(objectness, (width, height))
    plt.imshow(objectness / 255, alpha=0.5)
    plt.colorbar()
    return plot_to_image(figure, format=fmt)


def draw_classes_map(
    image: Union[np.ndarray, tf.Tensor],
    classes_map: Union[np.ndarray, tf.Tensor],
    figsize: Tuple[int, int] = (5, 5),
    score_threshold: float = 0.1,
    fmt: str = "png",
    subtitle: str = "",
    **kwargs,
) -> Image.Image:

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(classes_map, tf.Tensor):
        classes_map = classes_map.numpy()

    classes_map_shape = classes_map.shape
    fm_h, fm_w, num_classes = classes_map_shape
    if len(classes_map.shape) != 3:
        raise ValueError(
            "Classes map must be a tensor of shape " "[height, width, num_classes]"
        )

    label2color = {}
    if num_classes is not None:
        label2color = labels_to_colors(num_classes + 1)

    height, width = image.shape[:2]

    figure = plt.figure(figsize=figsize)
    plt.title(f"Classes map {classes_map_shape} {subtitle}")
    plt.imshow(image)

    color_map = np.zeros([fm_h, fm_w, 3])
    for j, i in itertools.product(range(fm_h), range(fm_w)):
        label = classes_map[j, i].argmax()
        score = classes_map[j, i].max()
        if score < score_threshold:
            label = num_classes

        color = label2color[label]
        color_map[j, i, :] = color

    color_map = (np.clip(color_map, 0, 1) * 255).astype("uint8")
    color_map = resize_nearest_neighbour(color_map, (width, height))
    plt.imshow(color_map / 255, alpha=0.5)
    return plot_to_image(figure, format=fmt)


def draw_classes_max_score_map(
    image: Union[np.ndarray, tf.Tensor],
    classes_map: Union[np.ndarray, tf.Tensor],
    figsize: Tuple[int, int] = (5, 5),
    score_threshold: float = 0.1,
    fmt: str = "png",
    subtitle: str = "",
    **kwargs,
) -> Image.Image:

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(classes_map, tf.Tensor):
        classes_map = classes_map.numpy()

    classes_map_shape = classes_map.shape
    fm_h, fm_w, num_classes = classes_map_shape
    if len(classes_map.shape) != 3:
        raise ValueError(
            "Classes map must be a tensor of shape " "[height, width, num_classes]"
        )

    height, width = image.shape[:2]
    figure = plt.figure(figsize=figsize)
    plt.title(f"Classes max score map {classes_map_shape} {subtitle}")
    plt.imshow(image)

    color_map = np.zeros([fm_h, fm_w])
    for j, i in itertools.product(range(fm_h), range(fm_w)):
        score = classes_map[j, i].max()
        if score < score_threshold:
            score = 0
        color_map[j, i] = score

    color_map = (np.clip(color_map, 0, 1) * 255).astype("uint8")
    color_map = resize_nearest_neighbour(color_map, (width, height))
    plt.imshow(color_map / 255, alpha=0.5)
    plt.colorbar()
    return plot_to_image(figure, format=fmt)


def draw_boxes(
    image: Union[np.ndarray, tf.Tensor],
    boxes_shape_map: Union[np.ndarray, tf.Tensor],
    objectness: Optional[Union[np.ndarray, tf.Tensor]] = None,
    classes_map: Optional[Union[np.ndarray, tf.Tensor]] = None,
    score_threshold: float = 0.3,
    figsize: Tuple[int, int] = (5, 5),
    fmt: str = "png",
    num_classes: Optional[int] = None,
    subtitle: str = "",
    fontsize: int = 15,
    linewidth: float = 2,
    **kwargs,
) -> Image.Image:

    if objectness is None:
        objectness = np.ones_like(boxes_shape_map)[..., 0]

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(objectness, tf.Tensor):
        objectness = objectness.numpy()
    if isinstance(boxes_shape_map, tf.Tensor):
        boxes_shape_map = boxes_shape_map.numpy()
    if classes_map is not None:
        if isinstance(classes_map, tf.Tensor):
            classes_map = classes_map.numpy()

    height, width = image.shape[:2]
    if len(boxes_shape_map.shape) == 3:
        boxes_shape_map = boxes_shape_map.reshape([-1, 4])
        assert len(objectness.shape) == 3 or len(objectness.shape) == 2
        objectness = objectness.reshape([-1])
        if classes_map is not None:
            num_classes = classes_map.shape[-1]
            classes_map = classes_map.argmax(-1).reshape([-1])
    elif len(boxes_shape_map.shape) == 2:
        assert len(objectness.shape) == 1
        assert objectness.shape[0] == boxes_shape_map.shape[0]
        if classes_map is not None:
            assert len(classes_map.shape) == 1
            assert classes_map.shape[0] == boxes_shape_map.shape[0]
    else:
        raise ValueError("Unsupported shape of input arrays!")

    indices = np.where(objectness > score_threshold)[0]
    label2color = {}
    if num_classes is not None:
        label2color = labels_to_colors(num_classes)

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111, aspect="equal")
    plt.title(f"Predictions {objectness.shape} {subtitle}")
    plt.imshow(image)

    for idx in indices:
        h, w, y, x = boxes_shape_map[idx] * np.array([height, width, height, width])
        pos = (x - w / 2, y - h / 2)
        color = "r"
        if classes_map is not None:
            label = classes_map[idx]
            color = label2color.get(label, color)
            plt.text(*pos, f"{label}", color=color, size=fontsize)
        rect = patches.Rectangle(
            pos, w, h, linewidth=linewidth, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

    return plot_to_image(figure, format=fmt)


def labels_to_colors(num_classes: int, brightness: float = 1) -> Dict[int, float]:
    hsv = [(i / max(num_classes - 1, 1), 1, brightness) for i in range(num_classes)]
    return {l: colorsys.hsv_to_rgb(*c) for l, c in enumerate(hsv)}


def draw_compare(
    target: Dict[str, Union[np.ndarray, tf.Tensor]],
    predicted: Dict[str, Union[np.ndarray, tf.Tensor]],
    draw_fn: Callable,
    **draw_kwargs,
) -> Image.Image:

    target_image = np.array(draw_fn(**target, **draw_kwargs, subtitle="target"))
    predicted_image = np.array(
        draw_fn(**predicted, **draw_kwargs, subtitle="predicted")
    )
    image = np.hstack([target_image, predicted_image])
    return Image.fromarray(image)


def draw_compares(
    target: Dict[str, Union[np.ndarray, tf.Tensor]],
    predicted: Optional[Dict[str, Union[np.ndarray, tf.Tensor]]],
    draw_fns: List[Callable],
    all_targets: bool = False,
    **draw_kwargs,
) -> Image.Image:

    images = []
    for k, draw_fn in enumerate(draw_fns):
        if k == 0 or all_targets:
            target_image = np.array(draw_fn(**target, **draw_kwargs, subtitle="target"))
            images.append(target_image)

        if predicted is not None:
            predicted_image = np.array(
                draw_fn(**predicted, **draw_kwargs, subtitle="predicted")
            )
            images.append(predicted_image)

    image = np.hstack(images)
    return Image.fromarray(image)


def draw_labels_frame_boxes(
    image: Union[np.ndarray, tf.Tensor],
    boxes: Union[np.ndarray, tf.Tensor],
    labels: Union[np.ndarray, tf.Tensor],
    figsize: Tuple[int, int] = (6, 6),
    title: Optional[str] = None,
    num_classes: Optional[int] = None,
    fontsize: int = 10,
    linewidth: float = 2,
    fmt: str = "png",
) -> Image:

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    if num_classes is None:
        num_classes = np.max(labels)

    num_boxes = boxes.shape[0]
    label2color = labels_to_colors(num_classes)
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111, aspect="equal")
    if title is not None:
        plt.title(title)

    if image.dtype != np.uint8 and image.max() > 2:
        image = image / 255.0

    plt.imshow(image)
    height, width = image.shape[:2]
    heights, widths = boxes_heights_widths(boxes.astype(np.float32))
    y_min, x_min = boxes[:, 0], boxes[:, 1]

    for idx in range(num_boxes):
        pos = (x_min[idx] * width, y_min[idx] * height)
        label = labels[idx]
        color = label2color.get(label, "r")
        plt.text(*pos, f"{label}", color=color, size=fontsize)
        rect = patches.Rectangle(
            pos,
            widths[idx] * height,
            heights[idx] * width,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    return plot_to_image(figure, format=fmt)
