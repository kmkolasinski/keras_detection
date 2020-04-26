from collections import defaultdict
from typing import List, Tuple, Callable, Optional, Any, Dict, Generator, Union, Type

import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from keras_detection.ops.tflite_ops import TFLiteModel
from keras_detection.structures import DataClass

keras = tf.keras
layers = tf.keras.layers


class DiffMetrics:
    AE = "ae"
    SE = "se"
    MAE = "mae"
    MSE = "mse"
    PER_CHANNEL_MAE = "per_channel_mae"
    PER_CHANNEL_MSE = "per_channel_mse"


@dataclass(frozen=True)
class OutputDiff(DataClass):
    left_name: str
    right_name: str
    shape: Tuple[int, ...]
    left_dtype: Type[np.dtype]
    right_dtype: Type[np.dtype]
    metrics: Dict[str, Union[List[np.ndarray], List[float], List[int]]]

    def append_metrics(self, diff: "OutputDiff") -> "OutputDiff":
        assert self.left_name == diff.left_name
        assert self.right_name == diff.right_name
        metrics = {k: v + diff.metrics[k] for k, v in self.metrics.items()}
        return self.replace(metrics=metrics)

    def as_flat_dict(self, agg: Callable[[List[np.ndarray]], float] = np.mean) -> Dict[str, Any]:
        """

        Args:
            agg: metrics aggregation method
        """
        metrics = {f"metric/{k}": agg(v) for k, v in self.metrics.items()}
        df_dict = self.to_dict()
        del df_dict["metrics"]
        df_dict.update(**metrics)
        return df_dict

    @staticmethod
    def to_df(diffs: List['OutputDiff'], agg: Callable[[List[np.ndarray]], float] = np.mean) -> Dict[str, List[Any]]:
        data = defaultdict(list)
        for diff in diffs:
            df = diff.as_flat_dict(agg=agg)
            for k, v in df.items():
                data[k].append(v)

        return data


def get_model_layers(model: keras.Model) -> List[layers.Layer]:
    """

    Args:
        model:

    Returns:

    """
    LAYERS_TO_SKIP = [layers.InputLayer]
    model_layers = []
    for layer in model.layers:
        if type(layer) in LAYERS_TO_SKIP:
            continue
        if isinstance(layer, keras.Model):
            model_layers += get_model_layers(layer)
        else:
            model_layers.append(layer)
    return model_layers


def convert_to_debug_model(model: keras.Model) -> keras.Model:

    model_layers = get_model_layers(model)

    # TODO when working with Quantized model for some
    #   reason Model(..., outputs=[...]) raises ValueError
    #   with info about disconnected graph.
    debug_model = None
    start_idx = 0
    while start_idx < len(model_layers):
        try:
            debug_model = keras.Model(
                inputs=model.inputs,
                outputs=[l.output for l in model_layers[start_idx:]],
                name="debug",
            )
            break
        except ValueError:
            print(
                f"Cannot create debug model. "
                f"Invalid layer at index {start_idx} '{model_layers[start_idx].name}'. Trying to skip it!"
            )
            start_idx += 1

    return debug_model


def match_debug_models_output_names(
    keras_model: keras.Model,
    tflite_model: TFLiteModel,
    name_match_fn: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, str]:

    assert isinstance(
        keras_model, keras.Model
    ), f"keras_model {keras_model} must be an instance of keras.Model"
    assert isinstance(
        tflite_model, TFLiteModel
    ), f"tflite_model {tflite_model} must be an instance of TFLiteModel"

    if name_match_fn is None:
        name_match_fn = lambda kn, tn: "/?/" in tn.replace(kn, "?")

    assert len(keras_model.output_names) == len(set(keras_model.output_names))
    assert len(tflite_model.output_names) == len(set(tflite_model.output_names))

    matches = {}
    for k, kname in enumerate(keras_model.output_names):
        for t, tname in enumerate(tflite_model.output_names):
            if name_match_fn(kname, tname):
                matches[kname] = tname
                break

    return matches


def diff_quantiztion_outputs(
    inputs: Any,
    keras_model: keras.Model,
    tflite_model: TFLiteModel,
    name_match_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[OutputDiff]:

    keras_outputs = keras_model.predict(inputs)
    keras_outputs = {n: o for n, o in zip(keras_model.output_names, keras_outputs)}
    tflite_outputs = tflite_model.predict(inputs)

    matches = match_debug_models_output_names(keras_model, tflite_model, name_match_fn)
    outputs_diffs = []

    for kname, tname in matches.items():
        ko = keras_outputs[kname]
        to = tflite_outputs[tname]
        left_dtype = ko.dtype
        right_dtype = to.dtype
        ko = np.array(ko, dtype=np.float64)
        to = np.array(to, dtype=np.float64)
        shape = ko.shape

        if len(shape) == 1:
            sh = [-1, 1, 1]
        elif len(shape) == 2:
            sh = [shape[0], 1, shape[1]]
        else:
            sh = [shape[0], -1, shape[1]]

        ko = ko.reshape(sh)
        to = to.reshape(sh)

        metrics = {
            DiffMetrics.MAE: np.abs(ko - to).mean(0).mean(),
            DiffMetrics.MSE: np.square(ko - to).mean(0).mean(),
            DiffMetrics.AE: np.abs(ko - to).mean(0).sum(),
            DiffMetrics.SE: np.square(ko - to).mean(0).sum(),
            DiffMetrics.PER_CHANNEL_MAE: np.abs(ko - to).mean(0).mean(0),
            DiffMetrics.PER_CHANNEL_MSE: np.square(ko - to).mean(0).mean(0),
        }

        diff = OutputDiff(
            left_name=kname,
            right_name=tname,
            shape=shape,
            left_dtype=left_dtype,
            right_dtype=right_dtype,
            metrics={k: [v] for k, v in metrics.items()},
        )
        outputs_diffs.append(diff)
    return outputs_diffs


def debug_models_quantization(
    representative_dataset: Generator,
    keras_model: keras.Model,
    tflite_model: TFLiteModel,
    name_match_fn: Optional[Callable[[str, str], bool]] = None,
    max_samples: Optional[int] = 16,
) -> List[OutputDiff]:

    outputs_diffs = []
    for k, inputs in enumerate(representative_dataset):
        if max_samples is not None and k >= max_samples:
            break

        batch_diff = diff_quantiztion_outputs(
            inputs, keras_model, tflite_model, name_match_fn
        )

        if len(outputs_diffs) == 0:
            outputs_diffs = batch_diff
        else:
            outputs_diffs = [
                prev.append_metrics(curr)
                for prev, curr in zip(outputs_diffs, batch_diff)
            ]

    return outputs_diffs


def debug_model_quantization(
    representative_dataset: Generator,
    keras_model: keras.Model,
    name_match_fn: Optional[Callable[[str, str], bool]] = None,
    max_samples: Optional[int] = 16,
) -> List[OutputDiff]:

    tflite_model = TFLiteModel.from_keras_model(keras_model)

    return debug_models_quantization(
        representative_dataset=representative_dataset,
        keras_model=keras_model,
        tflite_model=tflite_model,
        name_match_fn=name_match_fn,
        max_samples=max_samples
    )
