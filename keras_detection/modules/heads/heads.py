from typing import Optional, List

import tensorflow as tf

from keras_detection import FeatureMapPredictionTarget, FeatureMapDesc, ImageData
from keras_detection.losses import FeatureMapPredictionTargetLoss
from keras_detection.metrics import FeatureMapPredictionTargetMetric
from keras_detection.modules.core import Module

keras = tf.keras
LOGGER = tf.get_logger()


class Head(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_assigners: List[FeatureMapPredictionTarget] = []
        self.head_losses: List[FeatureMapPredictionTargetLoss] = []
        self.loss_weights: List[float] = []
        self.head_metrics: List[List[FeatureMapPredictionTargetMetric]] = [[]]
        self.fm_desc: Optional[FeatureMapDesc] = None

    def set_targets(self, targets: List[FeatureMapPredictionTarget]):
        self.target_assigners = targets

    def set_losses(
        self, losses: List[FeatureMapPredictionTargetLoss], weights: List[float]
    ):
        self.head_losses = losses
        self.loss_weights = weights

    def set_metrics(self, metrics: List[FeatureMapPredictionTargetMetric]):
        self.head_metrics = metrics

    def set_feature_map_description(self, fm_desc: FeatureMapDesc):
        self.fm_desc = fm_desc

    def compute_targets(self, batch: ImageData):
        return [
            ta.get_targets_tensors(self.fm_desc, batch.labels)
            for ta in self.target_assigners
        ]

    def compute_losses(
        self, targets: List[tf.Tensor], predictions: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        return [
            tf.constant(self.loss_weights[k]) * loss.call(targets[k], predictions[k])
            for k, loss in enumerate(self.head_losses)
        ]


class SingleConvHead(Head):
    def __init__(
        self,
        name: str,
        num_outputs: int,
        activation: Optional[str] = "relu",
        num_filters: int = 64,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_filters = num_filters
        self.activation = activation
        self.num_outputs = num_outputs
        self.conv1 = keras.layers.Conv2D(
            self.num_filters, kernel_size=3, padding="same"
        )
        self.conv1bn = keras.layers.BatchNormalization()
        self.conv1act = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(
            self.num_outputs, kernel_size=1, activation=None
        )

    def call(self, inputs: tf.Tensor, training: bool = None, mask=None):
        h = self.conv1(inputs)
        h = self.conv1bn(h)
        h = self.conv1act(h)
        h = self.conv2(h)
        if self.activation == "softmax":
            # to be compatible with tflite converter, otherwise
            # it will crash when exporting with representative_dataset
            h = keras.layers.Softmax()(h)
        else:
            h = keras.layers.Activation(self.activation)(h)

        return [h]
