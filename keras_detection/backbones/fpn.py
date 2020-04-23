from typing import Tuple, List

import tensorflow as tf

from keras_detection.backbones.base import Backbone

keras = tf.keras
K = tf.keras.backend
backend = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
keras_utils = tf.keras.utils


class FPNBackbone(Backbone):
    def __init__(self, backbone: Backbone, depth: int, num_first_blocks: int = 1):

        assert (
            backbone.num_fm_maps >= num_first_blocks
        ), "FPN cannot get more blocks than backbone have!"
        self.base_backbone = backbone
        self.num_first_blocks = num_first_blocks
        self.depth = depth

        shapes = [shape[1:] for shape in backbone.backbone.output_shape]
        self.fpn_backbone = build_fpn(shapes, depth)
        super().__init__(
            input_shape=backbone.input_shape, backbone=self.base_backbone.backbone
        )

    @property
    def num_fm_maps(self) -> int:
        return self.num_first_blocks

    def forward(
        self, inputs: tf.Tensor, is_training: bool = False, quantized: bool = False
    ) -> List[tf.Tensor]:
        outputs = self.backbone_forward(inputs, quantized=quantized)

        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]

        # TODO Check if we need to quantize FPN backbone
        outputs = self.fpn_backbone(outputs)
        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]

        return outputs[: self.num_first_blocks]


def build_fpn(
    feature_maps_shapes: List[Tuple[int, int, int]], depth: int
) -> keras.Model:
    feature_maps = [
        keras.layers.Input(shape=shape, name=f"fpn_input_{i}")
        for i, shape in enumerate(feature_maps_shapes)
    ]

    with tf.name_scope("fpn_top_down"):
        num_levels = len(feature_maps)
        outputs = []

        top_down = layers.Conv2D(
            depth, [1, 1], padding="same", name="projection_%d" % num_levels
        )(feature_maps[-1])
        outputs.append(top_down)

        for level in reversed(range(num_levels - 1)):
            top_down = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
                top_down
            )
            residual = layers.Conv2D(
                depth, [1, 1], padding="same", name="projection_%d" % (level + 1)
            )(feature_maps[level])
            top_down = 0.5 * top_down + 0.5 * residual
            # additional smoothing layer is used to mitigate the artifacts
            # appearing during up sampling
            outputs.append(
                layers.Conv2D(depth, [3, 3], padding="same", name="smoothing_%d" % (level + 1))(
                    top_down
                )
            )

    model = keras.models.Model(feature_maps, outputs[::-1], name="fpn")
    return model
