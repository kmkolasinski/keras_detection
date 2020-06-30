from dataclasses import dataclass
from typing import List, Dict, Any, Union, Tuple, Optional
from keras_detection.heads import Head, HeadFactory
from keras_detection.losses import FeatureMapPredictionTargetLoss
from keras_detection.metrics import FeatureMapPredictionTargetMetric
from keras_detection.structures import LabelsFrame
from keras_detection.targets.base import FeatureMapPredictionTarget, FeatureMapDesc

import tensorflow as tf


keras = tf.keras
MetricType = Union[keras.metrics.Metric, str, Any]
LossType = keras.losses.Loss


@dataclass
class PredictionTaskBase:
    name: str
    loss_weight: float
    target_builder: FeatureMapPredictionTarget
    loss: FeatureMapPredictionTargetLoss
    metrics: List[Union[FeatureMapPredictionTargetMetric, Any]]


@dataclass
class PredictionTaskDef(PredictionTaskBase):
    head_factory: HeadFactory


@dataclass
class PredictionTask(PredictionTaskBase):
    head: Head

    @classmethod
    def from_task_def(cls, head_output_name: str, task: PredictionTaskDef):
        return cls(
            name=task.name,
            loss_weight=task.loss_weight,
            target_builder=task.target_builder,
            loss=task.loss,
            metrics=task.metrics,
            head=task.head_factory.build(head_output_name),
        )

    def get_metrics(self, name_prefix: str = "") -> List[MetricType]:
        metrics = []
        for metric in self.metrics:
            if isinstance(metric, FeatureMapPredictionTargetMetric):
                metrics.append(metric.get_metric_fn(name_prefix))
            elif isinstance(metric, keras.metrics.Metric):
                if name_prefix not in metric.name and name_prefix != "":
                    metric._name = f"{name_prefix}/{metric._name}"
                metrics.append(metric)
            else:
                metrics.append(metric)
        return metrics

    def forward(
        self, feature_map: tf.Tensor, is_training: bool = False, quantized: bool = False
    ):
        return self.head.forward(
            feature_map, is_training=is_training, quantized=quantized
        )

    def postprocess(self, fm_desc: FeatureMapDesc, feature_map: tf.Tensor) -> tf.Tensor:
        return self.target_builder.postprocess_predictions(fm_desc, feature_map)


@dataclass
class FeatureMapPredictionTasks:
    name: str
    fm_desc: FeatureMapDesc
    tasks: List[PredictionTask]

    @property
    def outputs_names(self) -> List[str]:
        return [self.fm_task_name(t) for t in self.tasks]

    @property
    def targets_outputs_shapes(self) -> List[Tuple[Optional[int], int, int, int]]:
        shapes = []
        for t in self.tasks:
            nc = t.target_builder.num_outputs_with_weights
            sh = (None, self.fm_desc.fm_height, self.fm_desc.fm_width, nc)
            shapes.append(sh)
        return shapes

    def fm_task_name(self, task: PredictionTask) -> str:
        return f"{self.name}/{task.name}"

    def get_outputs(
        self, feature_map: tf.Tensor, is_training: bool = False, quantized: bool = False
    ) -> List[tf.Tensor]:

        outputs = []
        for t in self.tasks:
            output = t.forward(
                feature_map, is_training=is_training, quantized=quantized
            )
            outputs.append(output)
        return outputs

    def get_targets(self, frame: LabelsFrame[tf.Tensor]) -> Dict[str, tf.Tensor]:
        targets = {}
        for t in self.tasks:
            target = t.target_builder.get_targets_tensors(self.fm_desc, frame)
            targets[self.fm_task_name(t)] = target
        return targets

    def get_metrics(self) -> Dict[str, List[MetricType]]:
        return {self.fm_task_name(t): t.get_metrics() for t in self.tasks}

    def get_losses(self) -> Dict[str, LossType]:
        return {self.fm_task_name(t): t.loss for t in self.tasks}

    def get_losses_weights(self) -> Dict[str, float]:
        return {self.fm_task_name(t): t.loss_weight for t in self.tasks}
