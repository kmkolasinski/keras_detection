from typing import Tuple

from keras_detection.structures import FeatureMapDesc, LabelsFrame
import tensorflow as tf
import numpy as np
import keras_detection.datasets as datasets_ops


def maybe_enable_eager_mode():
    """This can be used when using TF version  >= 1.14"""
    if tf.__version__ < "2.0.0":
        tf.enable_eager_execution()


def create_fake_fm_desc(scale: int = 1):
    return FeatureMapDesc(
        fm_height=32 * scale,
        fm_width=24 * scale,
        image_height=64 * scale,
        image_width=48 * scale,
    )


def create_fake_boxframe(repeats: int = 1) -> LabelsFrame[tf.Tensor]:
    boxes = tf.constant(
        [[0.0, 0.0, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]] * repeats,
    )
    labels = tf.constant([1, 0, 0] * repeats, dtype=tf.int32)
    weights = tf.constant([0.5, 1.0, 0.0] * repeats)
    return LabelsFrame(boxes=boxes, labels=labels, weights=weights)


def create_fake_objectness_map(y_true: tf.Tensor, as_logits: bool = False) -> tf.Tensor:
    logits = tf.random.normal(tf.shape(y_true))
    if not as_logits:
        return tf.nn.sigmoid(logits)
    return logits


def create_fake_classes_map(y_true: tf.Tensor, as_logits: bool = False) -> tf.Tensor:
    logits = tf.random.normal(tf.shape(y_true))
    if not as_logits:
        print("logits:", logits.shape)
        return tf.nn.softmax(logits, axis=-1)
    return logits


def create_fake_boxes_map(y_true: tf.Tensor) -> tf.Tensor:
    boxes = tf.random.normal(tf.shape(y_true))
    return boxes


def create_fake_input_map(
    batch_size: int = 1, height: int = 32, width=24, num_channels: int = 5
) -> tf.Tensor:
    return tf.random.normal([batch_size, height, width, num_channels])


def create_fake_detection_dataset_generator(
    num_steps: int = 10, num_boxes: int = 5, num_classes: int = 10
):
    max_num_boxes = num_boxes
    for i in range(num_steps):
        num_boxes = np.random.randint(1, max_num_boxes)
        yx = np.random.rand(num_boxes, 2)
        hw = np.random.rand(num_boxes, 2) / 10
        ymin = yx[:, 0]
        xmin = yx[:, 1]
        ymax = yx[:, 0] + hw[:, 0]
        xmax = yx[:, 1] + hw[:, 1]
        boxes = np.stack([ymin, xmin, ymax, xmax]).T.astype(np.float32)
        labels = np.random.randint(0, num_classes, size=num_boxes)
        weights = np.ones_like(labels).astype(np.float32)
        image = np.random.randint(0, 255, size=(333, 777, 3), dtype=np.uint8)

        features = {"image": image}
        labels = {"boxes": boxes, "labels": labels, "weights": weights}
        yield {"features": features, "labels": labels}


def create_fake_detection_batched_dataset(
    image_size: Tuple[int, int] = (32, 24),
    num_epochs: int = 2,
    batch_size: int = 2,
    num_steps: int = 10,
    num_boxes: int = 5,
    num_classes: int = 10,
) -> tf.data.Dataset:

    dataset = datasets_ops.from_numpy_generator(
        create_fake_detection_dataset_generator(
            num_steps=num_steps, num_boxes=num_boxes, num_classes=num_classes
        )
    )

    def aug_fn(image_data):
        image = tf.cast(image_data.features.image, tf.float32)
        image = image / 255.0
        return image_data.replace_image(image)

    return datasets_ops.prepare_dataset(
        dataset,
        model_image_size=image_size,
        augmentation_fn=aug_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle_buffer_size=1,
        prefetch_buffer_size=1,
    )
