"""
Larger part of this code has be copied from https://github.com/qubvel/classification_models

The MIT License

Copyright (c) 2018, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import collections
from typing import Tuple, List, Any
import tensorflow as tf
from keras_detection.backbones.base import Backbone

keras = tf.keras
K = tf.keras.backend
backend = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
keras_utils = tf.keras.utils

ModelParams = collections.namedtuple(
    "ModelParams", ["model_name", "repetitions", "residual_block", "attention"]
)


class ResNetBackbone(Backbone):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_last_blocks: int = 1,
        units_per_block: Tuple[int, ...] = (1, 1),
        init_filters: int = 64,
        attention: Any = None,
        name: str = "ResNetBackbone",
    ):

        assert num_last_blocks <= len(units_per_block)

        self.units_per_block = units_per_block
        self.residual_conv_block = residual_conv_block
        self.attention = attention
        self.init_filters = init_filters
        self.num_last_blocks = num_last_blocks
        params = ModelParams(
            name, self.units_per_block, residual_conv_block, self.attention
        )
        backbone = CustomResNet(
            model_params=params, input_shape=input_shape, init_filters=init_filters,
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


def handle_block_names(stage, block):
    name_base = "stage{}_unit{}_".format(stage + 1, block + 1)
    conv_name = name_base + "conv"
    bn_name = name_base + "bn"
    relu_name = name_base + "relu"
    sc_name = name_base + "sc"
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        "kernel_initializer": "he_uniform",
        "use_bias": False,
        "padding": "valid",
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == "channels_last" else 1
    default_bn_params = {
        "axis": axis,
        "momentum": 0.99,
        "epsilon": 2e-5,
        "center": True,
        "scale": True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------


def residual_conv_block(
    filters, stage, block, strides=(1, 1), attention=None, cut="pre"
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        # BN alone is not supported by the tfmot
        # x = layers.BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = layers.Activation("relu", name=relu_name + "1")(input_tensor)

        # defining shortcut connection
        if cut == "pre":
            shortcut = input_tensor
        elif cut == "post":
            shortcut = layers.Conv2D(
                filters, (1, 1), name=sc_name, strides=strides, **conv_params
            )(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(
            filters, (3, 3), strides=strides, name=conv_name + "1", **conv_params
        )(x)

        x = layers.BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = layers.Activation("relu", name=relu_name + "2")(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + "2", **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(
    filters, stage, block, strides=None, attention=None, cut="pre"
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = layers.Activation("relu", name=relu_name + "1")(x)

        # defining shortcut connection
        if cut == "pre":
            shortcut = input_tensor
        elif cut == "post":
            shortcut = layers.Conv2D(
                filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params
            )(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv2D(filters, (1, 1), name=conv_name + "1", **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = layers.Activation("relu", name=relu_name + "2")(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(
            filters, (3, 3), strides=strides, name=conv_name + "2", **conv_params
        )(x)

        x = layers.BatchNormalization(name=bn_name + "3", **bn_params)(x)
        x = layers.Activation("relu", name=relu_name + "3")(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + "3", **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x

    return layer


def CustomResNet(
    model_params, init_filters=64, input_shape=None, input_tensor=None, **kwargs
):
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

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name="image")
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape, name="image")
        else:
            img_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block
    if model_params.attention:
        Attention = model_params.attention(**kwargs)
    else:
        Attention = None

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnet bottom
    x = layers.Conv2D(3, (1, 1), strides=(1, 1), name="conv0-pre", **conv_params)(img_input)
    x = layers.BatchNormalization(name="bn_data", **no_scale_bn_params)(x)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(
        init_filters, (7, 7), strides=(2, 2), name="conv0", **conv_params
    )(x)
    x = layers.BatchNormalization(name="bn0", **bn_params)(x)
    x = layers.Activation("relu", name="relu0")(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="pooling0")(x)

    # resnet body
    outputs = []
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    strides=(1, 1),
                    cut="post",
                    attention=Attention,
                )(x)

            elif block == 0:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    cut="post",
                    attention=Attention,
                )(x)

            else:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    strides=(1, 1),
                    cut="pre",
                    attention=Attention,
                )(x)

        outputs.append(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, outputs)
    return model
