import os
import warnings
from typing import Tuple, List, Optional

import tensorflow as tf
from keras_applications import correct_pad
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.mobilenet_v2 import (
    BASE_WEIGHT_PATH,
    _make_divisible,
    _inverted_res_block,
)
from keras_detection.backbones.base import Backbone

keras = tf.keras
K = tf.keras.backend
backend = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
keras_utils = tf.keras.utils
LOGGER = tf.get_logger()


class MobileNetV2Backbone(Backbone):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        alpha: float = 1.0,
        num_last_blocks: int = 1,
        min_fm_size: Optional[int] = None,
        weights: Optional[str] = None,
    ):
        self.num_last_blocks = num_last_blocks
        backbone = build_mobilenet_v2(
            input_shape=input_shape,
            alpha=alpha,
            weights=weights,
            input_tensor=None,
            min_fm_size=min_fm_size,
        )

        if len(backbone.outputs) < num_last_blocks:
            raise ValueError(
                "Cannot create Backbone, the number of requested num_last_blocks "
                f"({num_last_blocks}) if bigger than the number of resulting block "
                f"({len(backbone.outputs)}). Try do decrease min_fm_size value."
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
        return outputs[-self.num_last_blocks:]


def build_mobilenet_v2(
    input_shape=None,
    alpha=1.0,
    weights="imagenet",
    input_tensor=None,
    min_fm_size: Optional[int] = None,
):
    """Instantiates the MobileNetV2 architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper, but the name is kept for
        consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid alpha, rows when
            weights='imagenet'
    """

    import keras_applications.mobilenet_v2 as keras_mobilenet

    keras_mobilenet.backend = backend
    keras_mobilenet.layers = layers
    keras_mobilenet.models = models
    keras_mobilenet.keras_utils = keras_utils

    if min_fm_size is None:
        min_fm_size = 1

    if not (weights in {"imagenet", None} or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    keras_utils.get_source_inputs(input_tensor)
                )
            except ValueError:
                raise ValueError(
                    "input_tensor: ", input_tensor, "is not type input_tensor"
                )
        if is_input_t_tensor:
            if backend.image_data_format == "channels_first":
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "and input_tensor: ",
                        input_tensor,
                        "do not meet the same shape requirements",
                    )
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "and input_tensor: ",
                        input_tensor,
                        "do not meet the same shape requirements",
                    )
        else:
            raise ValueError(
                "input_tensor specified: ", input_tensor, "is not a keras tensor"
            )

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor: ",
                input_tensor,
                "is type: ",
                type(input_tensor),
                "which is not a valid type",
            )

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of `0.35`, `0.50`, `0.75`, "
                "`1.0`, `1.3` or `1.4` only."
            )

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            warnings.warn(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [96, 128, 160, 192, 224]."
                " Weights for input shape (224, 224) will be"
                " loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name="image")
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape, name="image")
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    first_block_filters = _make_divisible(32 * alpha, 8)

    outputs = []

    x = layers.ZeroPadding2D(
        padding=correct_pad(backend, img_input, 3), name="Conv1_pad"
    )(img_input)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name="Conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.99, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=16,
        alpha=alpha,
        stride=1,
        expansion=1,
        block_id=0,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=24,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=1,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=24,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=2,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=32,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=3,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=4,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=5,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=64,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=6,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=7,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=8,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=64,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=9,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=10,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=11,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=96,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=12,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=160,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=13,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=160,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=14,
    )
    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=160,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=15,
    )

    x = inverted_res_block(
        x,
        outputs,
        min_fm_size,
        filters=320,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=16,
    )
    print("outputs: ", outputs)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs, name="mobilenetv2_%0.2f_%s" % (alpha, rows))

    if weights is not None:
        LOGGER.info(f"Loading pre-trained model weights: {weights}")

    # Load weights.
    if weights == "imagenet":
        model_name = (
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
            + str(alpha)
            + "_"
            + str(rows)
            + "_no_top"
            + ".h5"
        )
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras_utils.get_file(
            model_name, weight_path, cache_subdir="models"
        )
        model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    elif weights is not None:
        model.load_weights(weights, skip_mismatch=True)

    return model


def inverted_res_block(x, outputs, min_fm_size, **kwargs):
    if x.shape[1] == min_fm_size:
        return x

    if x.shape[1] // kwargs.get("stride") >= min_fm_size:
        x = _inverted_res_block(x, **kwargs)
        print(f" > stride={kwargs.get('stride')} shape={x.shape}")
        if kwargs.get("stride") == 2:
            outputs.append(x)
    return x
