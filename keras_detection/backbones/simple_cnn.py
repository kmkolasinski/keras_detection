from typing import Tuple, List
import tensorflow as tf
from keras_detection.backbones.base import Backbone

keras = tf.keras
K = tf.keras.backend


class SimpleCNNBackbone(Backbone):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_last_blocks: int = 1,
        num_blocks: int = 2,
        init_filters: int = 64,
        name: str = "SimpleCNNBackbone",
    ):

        assert num_last_blocks <= num_blocks
        self.num_blocks = num_blocks
        self.init_filters = init_filters
        self.num_last_blocks = num_last_blocks

        backbone = simple_cnn(
            input_shape=input_shape,
            init_filters=init_filters,
            num_blocks=num_blocks,
            name=name,
        )
        super().__init__(input_shape=input_shape, backbone=backbone)

    @property
    def num_fm_maps(self) -> int:
        return self.num_last_blocks

    def forward(
        self, inputs: tf.Tensor, is_training: bool = False, quantized: bool = False
    ) -> List[tf.Tensor]:
        outputs = self.backbone_forward(inputs, quantized=quantized)
        if isinstance(outputs, tf.Tensor):
            return [outputs]

        return outputs[-self.num_last_blocks :]


def simple_cnn(
    input_shape: Tuple[int, int, int], init_filters: int, num_blocks: int, name: str
) -> keras.Model:

    img_input = keras.layers.Input(shape=input_shape, name="image")
    h = img_input
    h = keras.layers.Conv2D(init_filters, (3, 3), padding="same", strides=(2, 2))(h)
    h = keras.layers.Activation("relu")(h)

    outputs = []
    for i in range(num_blocks):
        n = init_filters * (i + 1)
        h = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(h)
        h = keras.layers.Conv2D(n, (3, 3), padding="same")(h)
        h = keras.layers.Activation("relu")(h)
        h = keras.layers.Conv2D(n, (3, 3), padding="same")(h)
        h = keras.layers.Activation("relu")(h)
        h = keras.layers.Dropout(0.25)(h)
        outputs.append(h)

    model = keras.models.Model(img_input, outputs, name=name)
    return model
