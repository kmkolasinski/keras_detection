from typing import List, Callable
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization as BatchNormalizationV2
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import (
    QuantizeWrapper,
)

keras = tf.keras
LOGGER = tf.get_logger()


def get_all_bn_layers(layers: List[keras.layers.Layer]) -> List[keras.layers.Layer]:
    bn_layers = []
    for layer in layers:

        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
        elif isinstance(layer, QuantizeWrapper):
            if isinstance(layer.layer, BatchNormalization):
                bn_layers.append(layer)
            elif isinstance(layer.layer, BatchNormalizationV2):
                bn_layers.append(layer)
        elif isinstance(layer, keras.Model):
            bn_layers += get_all_bn_layers(layer.layers)
    return bn_layers


def freeze_model_bn_layers(model: keras.Model, trainable: bool = True):
    bn_layers = get_all_bn_layers(model.layers)
    for bn_layer in bn_layers:
        bn_layer.trainable = trainable

    msg = "Freezed" if not trainable else "Unfreezed"
    LOGGER.info(f"{msg} N={len(bn_layers)} BatchNormalization layers")


def get_l2_loss_fn(
    model: keras.Model, l2_reg: float = 1e-6, skip_pattern: str = "beta"
) -> Callable[[], tf.Tensor]:
    """
    Return L2 loss for trainable weights.

    TODO(kkol): explicitly state which layers are omitted,
        make skip pattern a list

    Args:
        model: keras Model
        l2_reg:

    Returns:
        scalar loss tensor
    """

    def l2_regularization():
        weights = []
        for w in model.weights:
            if w.trainable and skip_pattern not in w.name:
                weights.append(w)

        return l2_reg * tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])

    return l2_regularization
