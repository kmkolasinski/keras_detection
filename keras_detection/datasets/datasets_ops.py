from typing import Dict, Optional, Callable, Tuple
import tensorflow as tf
from keras_detection.structures import ImageData, get_padding_shapes
from keras_detection.ops.python_ops import NestedDict
import keras_detection.ops.python_ops as py_ops

ImageDataMapFn = Callable[[ImageData[tf.Tensor]], ImageData[ImageData[tf.Tensor]]]


def prepare_dataset(
    dataset: tf.data.Dataset,
    model_image_size: Tuple[int, int],
    augmentation_fn: Optional[ImageDataMapFn] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int] = None,
    num_parallel_calls: Optional[int] = None,
    prefetch_buffer_size: Optional[int] = None,
    prefetch_to_device: Optional[str] = None,
) -> tf.data.Dataset:

    # apply data augmentation:
    if augmentation_fn is not None:
        dataset = dataset.map(
            map_image_data(augmentation_fn), num_parallel_calls=num_parallel_calls,
        )

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(
        map_image_data(prepare_for_batching(model_image_size)),
        num_parallel_calls=num_parallel_calls,
    )

    # batching and padding
    if batch_size is not None:
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=get_padding_shapes(
                dataset, spatial_image_shape=model_image_size
            ),
            drop_remainder=True,
        )

    # try to prefetch dataset on certain device
    if prefetch_to_device is not None:
        prefetch_fn = tf.data.experimental.prefetch_to_device(
            device=prefetch_to_device, buffer_size=prefetch_buffer_size
        )
        dataset = dataset.apply(prefetch_fn)
    else:
        if prefetch_buffer_size is not None:
            dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset


def from_numpy_generator(generator) -> tf.data.Dataset:

    example: NestedDict = next(generator)
    output_types = py_ops.map_nested_dict(example, lambda e: tf.as_dtype(e.dtype))

    # TODO fix output shapes to be defined, here only the last
    #   dimension is not None
    output_shapes = py_ops.map_nested_dict(
        example,
        lambda e: tf.TensorShape([None] * (len(e.shape) - 1) + [e.shape[-1]])
        if len(e.shape) > 1
        else [None],
    )
    return tf.data.Dataset.from_generator(
        lambda: generator, output_types=output_types, output_shapes=output_shapes
    )


def map_image_data(
    map_fn: ImageDataMapFn,
) -> Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]:
    def map_function(image_data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        data = ImageData.from_dict(image_data)
        data = map_fn(data)
        return data.to_dict()

    return map_function


def prepare_for_batching(size: Tuple[int, int]) -> ImageDataMapFn:
    def map_fn(data: ImageData) -> ImageData:
        image = tf.image.resize(data.features.image, size)
        data = data.replace_image(image=image)
        if data.has_labels():
            num_rows = tf.shape(data.labels.boxes)[0]
            data = data.replace_frame_field("num_rows", num_rows)
        return data

    return map_fn
