"""
Simple detection metrics
"""
from collections import defaultdict
from typing import List

from dataclasses import dataclass

from keras_detection.structures import DataClass, LabelsFrame
from keras_detection.ops import np_frame_ops
import numpy as np


class MetricNames:
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"


@dataclass(frozen=True)
class Metric(DataClass):
    name: str
    value: float
    weight: float


def aggregate_metrics(metrics: List[Metric]) -> List[Metric]:
    """
    Aggregate metrics by their names and compute weighted mean
    Args:
        metrics:


    """
    metrics_by_name = defaultdict(list)
    for m in metrics:
        metrics_by_name[m.name].append(m)

    new_metrics = []
    for name, metrics in metrics_by_name.items():
        values = [m.value for m in metrics]
        weights = [m.weight for m in metrics]
        mean_value = np.average(values, weights=weights)
        new_metrics.append(Metric(name, mean_value, np.sum(weights)))

    return new_metrics


def image_localization_metrics(
    target: LabelsFrame[np.ndarray],
    predicted: LabelsFrame[np.ndarray],
    iou_threshold: float = 0.35,
) -> List[Metric]:
    """
    Per image localization precision recall and f1_score metrics
    Args:
        target:
        predicted:
        iou_threshold:

    Returns:
        precision, recall, f1_score metrics values
    """

    assert isinstance(target, LabelsFrame), "Targets frame must be an instance of LabelsFrame"
    assert isinstance(predicted, LabelsFrame), "Predictions frame must be an instance of LabelsFrame"

    t_indices, p_indices = np_frame_ops.argmax_iou_matching(
        target.boxes, predicted.boxes, iou_threshold
    )

    num_matches = np.sum(t_indices > -1)
    num_targets = target.boxes.shape[0]
    num_predictions = predicted.boxes.shape[0]

    precision = num_matches / max(num_predictions, 1)
    recall = num_matches / max(num_targets, 1)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    prefix = f"localization@{iou_threshold}/"
    return [
        Metric(prefix + MetricNames.PRECISION, precision, num_targets),
        Metric(prefix + MetricNames.RECALL, recall, num_targets),
        Metric(prefix + MetricNames.F1_SCORE, f1_score, num_targets),
    ]
