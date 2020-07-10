from typing import Tuple, Any
from keras_detection.backbones.resnet import ResNetBackbone
from keras_detection.modules.core import Module
import tensorflow as tf


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
        **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.backbone_model = ResNetBackbone(
            input_shape=input_shape,
            num_last_blocks=num_last_blocks,
            units_per_block=units_per_block,
            init_filters=init_filters,
            attention=attention,
            name=name,
        ).backbone
        self.num_last_blocks = num_last_blocks
        self.build(input_shape=(None, *input_shape))

    def call(self, inputs, training=None, mask=None):
        outputs = self.backbone_model(inputs)
        if isinstance(outputs, tf.Tensor):
            return [outputs]
        return outputs[-self.num_last_blocks:]
