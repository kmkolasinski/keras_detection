from typing import Tuple, Any, List

import tensorflow as tf

from keras_detection import FeatureMapDesc
from keras_detection.backbones.resnet import ResNetBackbone
from keras_detection.modules.core import Module


class ResNet(Module):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_last_blocks: int = 1,
        units_per_block: Tuple[int, ...] = (1, 1),
        init_filters: int = 64,
        attention: Any = None,
        name: str = "ResNetBackbone",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.backbone = ResNetBackbone(
            input_shape=input_shape,
            num_last_blocks=num_last_blocks,
            units_per_block=units_per_block,
            init_filters=init_filters,
            attention=attention,
            name=name,
        ).backbone
        self.num_last_blocks = num_last_blocks
        # Initialize output and shapes
        self(tf.keras.Input(shape=input_shape, name="image"))

    def get_output_shapes(self) -> List[Tuple[int, int, int]]:
        shapes = self.output_shape
        return [shapes[f"fm{k}"][1:] for k in range(len(shapes))]

    def get_output_names(self, name: str) -> List[str]:
        shapes = self.output_shape
        return [f"{name}/fm{k}" for k in range(len(shapes))]

    def call(self, inputs, training=None, mask=None):
        outputs = self.backbone(inputs)
        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]
        outputs = outputs[-self.num_last_blocks :]

        named_outputs = {}
        for fm_id, o in enumerate(outputs):
            named_outputs[f"fm{fm_id}"] = o
        return named_outputs


class FeatureMapDescEstimator:
    def __call__(self, image: tf.Tensor, feature_map: tf.Tensor, **kwargs):
        fm_desc = FeatureMapDesc(
            *feature_map.shape[1:3].as_list(), *image.shape[1:3].as_list()
        )
        return fm_desc
