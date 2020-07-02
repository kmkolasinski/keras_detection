from abc import abstractmethod, ABC
from typing import Optional, Tuple, Type

import tensorflow as tf
import tensorflow_model_optimization as tfmo

from keras_detection.utils.dvs import *

keras = tf.keras
LOGGER = tf.get_logger()


class Head(ABC):
    def __init__(self, output_name: str, num_outputs: int, **kwargs):
        super().__init__(**kwargs)
        self._num_outputs = num_outputs
        self._output_name = output_name
        self._head_model: Optional[keras.Model] = None

    def get_head_name(self, input_shape: Tuple[int, int, int]) -> str:
        height, width = input_shape[:2]
        return f"head/fm{height}x{width}/{self.output_name}"

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def num_outputs(self) -> int:
        """Num output filters"""
        return self._num_outputs

    @abstractmethod
    def build(
        self, input_shape: Tuple[int, int, int], is_training: bool = False
    ) -> keras.Model:
        pass

    def forward(
        self,
        feature_map: (B, H, W, C),
        is_training: bool = False,
        quantized: bool = False,
    ) -> tf.Tensor:

        input_shape = feature_map.shape[1:]
        if self._head_model is None:
            self._head_model = self.build(input_shape, is_training=is_training)

        if not quantized:
            outputs = self._head_model(feature_map)
        else:
            LOGGER.info(f"Running quantization for head: '{self.output_name}'")
            model = tfmo.quantization.keras.quantize_model(self._head_model)
            outputs = model(feature_map)

        return outputs


class HeadFactory:
    def __init__(self, htype: Type[Head], **kwargs):
        self._kwargs = kwargs
        self._htype = htype

    def build(self, output_name: str) -> Head:
        return self._htype(output_name=output_name, **self._kwargs)


class ActivationHead(Head):
    def __init__(self, output_name: str, num_outputs: int, activation: str = None):
        super().__init__(output_name=output_name, num_outputs=num_outputs)
        self.activation = activation

    def build(
        self, input_shape: Tuple[int, int, int], is_training: bool = False
    ) -> keras.Model:
        x = keras.layers.Input(shape=input_shape)
        return keras.Model(x, x, name=self.get_head_name(input_shape))

    def forward(
        self,
        feature_map: (B, H, W, C),
        is_training: bool = False,
        quantized: bool = False,
    ) -> tf.Tensor:

        h = keras.layers.Activation(self.activation, name=self.output_name)(feature_map)
        return h


class SingleConvHead(Head):
    def __init__(
        self,
        output_name: str,
        num_outputs: int,
        activation: Optional[str] = "relu",
        num_filters: int = 64,
    ):
        super().__init__(output_name=output_name, num_outputs=num_outputs)
        self._num_filters = num_filters
        self._activation = activation

    def build(
        self, input_shape: Tuple[int, int, int], is_training: bool = False
    ) -> keras.Model:

        x = keras.layers.Input(shape=input_shape)
        h = keras.layers.Conv2D(self._num_filters, kernel_size=3, padding="same")(x)
        h = keras.layers.BatchNormalization()(h)
        h = keras.layers.ReLU()(h)
        h = keras.layers.Conv2D(self.num_outputs, kernel_size=1, activation=None)(h)
        return keras.Model(x, h, name=self.get_head_name(input_shape))

    def forward(
        self,
        feature_map: (B, H, W, C),
        is_training: bool = False,
        quantized: bool = False,
    ) -> tf.Tensor:
        h = super().forward(feature_map, is_training, quantized)

        if self._activation == "softmax":
            # to be compatible with tflite converter, otherwise
            # it will crash when exporting with representative_dataset
            h = keras.layers.Softmax(name=self.output_name)(h)
        else:
            h = keras.layers.Activation(self._activation, name=self.output_name)(h)

        return h


class SingleConvPoolHead(SingleConvHead):

    def build(
        self, input_shape: Tuple[int, int, int], is_training: bool = False
    ) -> keras.Model:
        x = keras.layers.Input(shape=input_shape)
        h = keras.layers.Conv2D(self._num_filters, kernel_size=3, padding="same")(x)
        h = keras.layers.BatchNormalization()(h)
        # h = keras.layers.ReLU()(h)
        h = keras.layers.GlobalAveragePooling2D()(h)
        h = keras.layers.Dense(self.num_outputs, activation=None)(h)
        h = keras.layers.Reshape([1, 1, self.num_outputs])(h)
        return keras.Model(x, h, name=self.get_head_name(input_shape))


class NoQuantizableSingleConvHead(SingleConvHead):
    def forward(
        self,
        feature_map: (B, H, W, C),
        is_training: bool = False,
        quantized: bool = False,
    ) -> tf.Tensor:
        return super().forward(feature_map, is_training, quantized=False)


class SingleConvHeadFactory(HeadFactory):
    def __init__(
        self,
        num_outputs: int,
        activation: Optional[str] = "relu",
        num_filters: int = 64,
        htype: Type[SingleConvHead] = SingleConvHead,
    ):
        super().__init__(
            htype,
            num_outputs=num_outputs,
            activation=activation,
            num_filters=num_filters,
        )


class SingleConvPoolHeadFactory(SingleConvHeadFactory):
    pass