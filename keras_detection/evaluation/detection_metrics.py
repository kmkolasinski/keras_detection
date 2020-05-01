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


def _precision_recall_f1_score(
    prefix: str, num_matches: int, num_targets: int, num_predictions: int
) -> List[Metric]:

    precision = num_matches / max(num_predictions, 1)
    recall = num_matches / max(num_targets, 1)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    return [
        Metric(prefix + MetricNames.PRECISION, precision, num_targets),
        Metric(prefix + MetricNames.RECALL, recall, num_targets),
        Metric(prefix + MetricNames.F1_SCORE, f1_score, num_targets),
    ]


def image_precision_recall_metrics(
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

    # TODO Allow use weights from targets
    if target.weights is not None:
        raise NotImplementedError(
            "Weighted metrics are not implemented, yet. Set targets.weights to None "
            "to skip this error."
        )

    assert isinstance(
        target, LabelsFrame
    ), "Targets frame must be an instance of LabelsFrame"
    assert isinstance(
        predicted, LabelsFrame
    ), "Predictions frame must be an instance of LabelsFrame"

    t_indices, p_indices = np_frame_ops.argmax_iou_matching(
        target.boxes, predicted.boxes, iou_threshold
    )

    # only valid matches
    valid_t_indices = np.where(t_indices > -1)[0]

    num_matches = valid_t_indices.shape[0]
    num_targets = target.boxes.shape[0]
    num_predictions = predicted.boxes.shape[0]

    loc_metrics = _precision_recall_f1_score(
        f"localization@{iou_threshold}/", num_matches, num_targets, num_predictions
    )

    t_labels = target.labels[valid_t_indices]
    p_labels = predicted.labels[t_indices[valid_t_indices]]

    num_matches = int(np.sum(t_labels == p_labels))
    num_targets = target.boxes.shape[0]
    num_predictions = predicted.boxes.shape[0]

    det_metrics = _precision_recall_f1_score(
        f"detection@{iou_threshold}/", num_matches, num_targets, num_predictions
    )

    return loc_metrics + det_metrics
