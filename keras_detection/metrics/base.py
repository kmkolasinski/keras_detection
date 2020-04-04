from abc import abstractmethod, ABC
from dataclasses import dataclass
import tensorflow as tf
from tensorflow_addons.metrics import MeanMetricWrapper
from keras_detection.targets.base import FeatureMapPredictionTarget

keras = tf.keras


@dataclass(frozen=True)
class FeatureMapPredictionTargetMetric(ABC):
    target_def: FeatureMapPredictionTarget

    def get_metric_fn(self, name_prefix: str = "") -> MeanMetricWrapper:
        def metric_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
            y_true, weights = self.target_def.to_targets_and_weights(y_true)
            if weights is None:
                fm_height, fm_width = y_pred.shape[1:2]
                weights = tf.ones([1, fm_height, fm_width, 1])
            return self.compute(y_true, y_pred, weights)

        name = f"{name_prefix}/{self.name}"
        if name_prefix == "":
            name = self.name

        return MeanMetricWrapper(metric_fn, name)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def compute(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        pass
