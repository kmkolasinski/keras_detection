from abc import ABC, abstractmethod
from typing import Tuple, List, Any

import tensorflow as tf
import tensorflow_model_optimization as tfmo

keras = tf.keras
LOGGER = tf.get_logger()


class Backbone(ABC):
    def __init__(self, backbone: keras.Model, input_shape: Tuple[int, int, int]):
        self._image_input_shape = input_shape
        self._backbone: keras.Model = backbone

    def get_input_tensor(self) -> tf.Tensor:
        return self._backbone.input

    @property
    def backbone(self) -> keras.Model:
        return self._backbone

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._image_input_shape

    @property
    @abstractmethod
    def num_fm_maps(self) -> int:
        pass

    def preprocess_images(
        self, inputs: tf.Tensor, is_training: bool = False
    ) -> tf.Tensor:
        return tf.cast(inputs, tf.float32) / 255.0

    def backbone_forward(self, inputs: tf.Tensor, quantized: bool = False) -> Any:
        if not quantized:
            outputs = self._backbone(inputs)
        else:
            LOGGER.info(
                f"Running quantization for model backbone: {self._backbone.name}"
            )
            _backbone = tfmo.quantization.keras.quantize_model(self._backbone)
            outputs = _backbone(inputs)
        return outputs

    @abstractmethod
    def forward(
        self, inputs: tf.Tensor, is_training: bool = False, quantized: bool = False
    ) -> List[tf.Tensor]:
        pass
