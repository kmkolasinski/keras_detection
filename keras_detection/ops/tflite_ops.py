from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Union, Dict, Optional, Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# See https://github.com/tensorflow/tensorflow/issues/32180
# once this issue will be fixed I can use tf.lite.TFLiteConverter.from_keras_model
from keras_detection.utils.tflite_converter_shitfix import from_keras_model

LOGGER = tf.get_logger()
keras = tf.keras


# No quantization, float32
TFLITE_F32_SUFFIX = ".f32.tflite"
# Dynamic range quantized
TFLITE_DR_SUFFIX = ".dr.quantized.tflite"
# Fixed range quantized
TFLITE_FR_SUFFIX = ".fr.quantized.tflite"


class TFLiteModel:
    def __init__(self, tflite_model: Union[bytes, str, Path]):
        self.tflite_model = tflite_model
        interpreter, predict_fn = create_tflite_predict_fn(tflite_model)
        self.interpreter = interpreter
        self.predict_fn = predict_fn

    @classmethod
    def from_keras_model(
        cls,
        model: keras.Model,
        optimizations: Optional[List[tf.lite.Optimize]] = None,
        dataset: Optional[tf.data.Dataset] = None,
        num_samples: int = 32,
    ) -> "TFLiteModel":
        if dataset is None:
            return cls(convert_default(model, optimizations=optimizations))
        else:
            return cls(
                convert_quantized(
                    model,
                    dataset=dataset,
                    num_samples=num_samples,
                    optimizations=optimizations,
                )
            )

    def get_details(self, inputs: bool, key: str) -> List[Any]:
        if inputs:
            details = self.interpreter.get_input_details()
        else:
            details = self.interpreter.get_output_details()
        return [d[key] for d in details]

    @property
    def output_names(self) -> List[str]:
        return self.get_details(inputs=False, key="name")

    @property
    def input_names(self) -> List[str]:
        return self.get_details(inputs=True, key="name")

    @property
    def output_shapes(self) -> List[np.ndarray]:
        return self.get_details(inputs=False, key="shape")

    @property
    def input_shapes(self) -> List[np.ndarray]:
        return self.get_details(inputs=True, key="shape")

    @property
    def output_dtypes(self) -> List[np.dtype]:
        return self.get_details(inputs=False, key="dtype")

    @property
    def input_dtypes(self) -> List[np.dtype]:
        return self.get_details(inputs=True, key="dtype")

    def validate_inputs(self, inputs: List[np.ndarray]):
        input_details = self.interpreter.get_input_details()
        validated_inputs = []
        if len(input_details) != len(inputs):
            raise ValueError(
                f"Invalid number of input arguments. Expected {len(inputs)}, "
                f"but got {len(input_details)}. Model input details: {input_details}"
            )
        for v, detail in zip(inputs, input_details):
            v = np.array(v)

            if v.dtype == np.float64 and detail["dtype"] == np.float32:
                v = v.astype(np.float32)

            if list(v.shape) != detail["shape"].tolist():
                raise ValueError(
                    f"Invalid input shape for '{detail['name']}'. "
                    f"Expected {detail['shape']}, but got: {v.shape}"
                )
            if v.dtype != detail["dtype"]:

                raise ValueError(
                    f"Invalid input dtype for '{detail['name']}'. "
                    f"Expected {detail['dtype']}, but got: {v.dtype}"
                )

            validated_inputs.append(v)
        return validated_inputs

    def predict(self, *inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = self.validate_inputs(inputs)
        return self.predict_fn(inputs)

    def test_predict(self, num_test_steps: int = 1) -> None:
        """Run test predict on random inputs to smoke test converted model"""
        test_run_tflite_model(self.tflite_model, num_test_steps=num_test_steps)

    def dump(self, save_path: Path) -> Path:
        LOGGER.info(f"Saving TFLite model: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as file:
            file.write(self.tflite_model)
        return save_path


def convert_default(
    model: keras.Model, optimizations: Optional[List[tf.lite.Optimize]] = None
) -> bytes:

    if optimizations is None:
        optimizations = [tf.lite.Optimize.DEFAULT]

    LOGGER.info(
        f"Building TFLite model from Keras Model using optimizations: {optimizations}"
    )
    converter = from_keras_model(model)
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.optimizations = optimizations

    return converter.convert()


def convert_quantized(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_samples: int = 32,
    optimizations: Optional[List[tf.lite.Optimize]] = None,
) -> bytes:

    if optimizations is None:
        optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for k, data in enumerate(dataset):

            if isinstance(data, tuple) and len(data) == 2:
                features = data[0]
                if not isinstance(features, dict):
                    raise ValueError(
                        f"Features must be a dict, got {list(features)} with values: {features}"
                    )
            elif isinstance(data, dict):
                features = data
            else:
                raise ValueError(
                    "Representative dataset should return (features, labels) tuple or "
                    "single dict with features inputs"
                )

            if k >= num_samples:
                break
            LOGGER.debug(f"Sampling {k + 1}/{num_samples}")
            yield [features[name] for name in model.input_names]

    converter = from_keras_model(model)
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.optimizations = optimizations
    converter.target_spec.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen

    return converter.convert()


def create_tflite_predict_fn(tflite_model: Union[bytes, str, Path]):
    if type(tflite_model) == bytes:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
    elif isinstance(tflite_model, Path) or isinstance(tflite_model, str):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
    else:
        raise ValueError(
            f"tflite_model must be one of [bytes, str, Path], got: {type(tflite_model)}"
        )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict_fn(inputs: List[np.ndarray]):
        assert len(inputs) == len(input_details), (
            f"Invalid number of inputs, expecting"
            f" {len(input_details)} input tensors."
        )
        for inp, info in zip(inputs, input_details):
            interpreter.set_tensor(info["index"], inp)
        interpreter.invoke()
        outputs = {}
        for info in output_details:
            output = interpreter.get_tensor(info["index"])
            outputs[info["name"]] = output
        return outputs

    return interpreter, predict_fn


def test_run_tflite_model(
    tflite_model: Union[bytes, str, Path], num_test_steps: int = 1
) -> None:
    LOGGER.info(f"Testing tflite model for N={num_test_steps} iterations ..")
    interpreter, predict_fn = create_tflite_predict_fn(tflite_model)
    print(f"Input details:")
    for info in interpreter.get_input_details():
        pprint(info, indent=2)
    print(f"Output details:")
    for info in interpreter.get_output_details():
        pprint(info, indent=2)
    start = datetime.now()
    for _ in range(num_test_steps):
        inputs = []
        for input_detail in interpreter.get_input_details():
            input_shape = input_detail["shape"]
            dtype = input_detail["dtype"]
            random_input = np.random.rand(*input_shape).astype(dtype)
            inputs.append(random_input)
        predict_fn(inputs)

    delta = datetime.now() - start
    LOGGER.info(f"Test finished in {delta.total_seconds()} seconds")


def convert_model_to_tflite(
    model: keras.Model,
    save_path: Union[str, Path],
    dataset: Optional[tf.data.Dataset] = None,
    num_dataset_samples: int = 16,
) -> List[Path]:

    output_files = []
    save_path = Path(save_path)

    LOGGER.info(f"Exporting model to TFLite: {save_path}")
    try:
        tflite_f32_model = TFLiteModel.from_keras_model(model, optimizations=[])
        tflite_f32_model.test_predict(1)
        path = tflite_f32_model.dump(save_path.with_suffix(TFLITE_F32_SUFFIX))
        output_files.append(path)
    except RuntimeError as error:
        LOGGER.warning("Cannot generate float32 model. Probably you are trying to "
                       "convert model with quantized (or fake quantized) nodes. "
                       f"Error: {error}")

    tflite_dr_model = TFLiteModel.from_keras_model(model)
    tflite_dr_model.test_predict(1)
    path = tflite_dr_model.dump(save_path.with_suffix(TFLITE_DR_SUFFIX))
    output_files.append(path)

    if dataset is not None:
        LOGGER.info(f"Converting quantized model")
        # checking dataset batch size and model
        features_shapes = tf.compat.v1.data.get_output_shapes(dataset)[0]
        dataset_input_shapes = {
            name: shape[0] for name, shape in features_shapes.items()
        }
        model_input_shapes = {
            name: tensor.shape[0] or 1
            for name, tensor in zip(model.input_names, model.inputs)
        }
        for name, bs in model_input_shapes.items():
            msg = (
                "Dataset and model must have the same batch size. If "
                "exported model have batch_size = None, then dataset "
                f"should be set to batch_size = 1. Input {name} has "
                f"shape {dataset_input_shapes[name]}, but expected batch "
                f"size to be equal {bs}."
            )
            assert dataset_input_shapes[name] == bs, msg

        tflite_fr_model = TFLiteModel.from_keras_model(
            model, dataset=dataset, num_samples=num_dataset_samples
        )
        tflite_fr_model.test_predict(1)
        path = tflite_fr_model.dump(save_path.with_suffix(TFLITE_FR_SUFFIX))
        output_files.append(path)

    return output_files


def verify_converted_model_output_statistics(
    dataset: tf.data.Dataset,
    keras_model: keras.Model,
    tflite_model: Path,
    num_test_steps: int,
) -> Dict[str, Dict[str, List[float]]]:

    dataset = iter(dataset)  # make dataset iterable
    LOGGER.info(f"Testing converted model: {tflite_model.name}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # measure MAE and MSE between keras model and tflite
    metrics = defaultdict(lambda: defaultdict(list))

    for k, (features, _) in tqdm(enumerate(dataset), total=num_test_steps):
        if k >= num_test_steps:
            break
        predictions = keras_model.predict(features)
        if isinstance(predictions, tf.Tensor) or isinstance(predictions, np.ndarray):
            # keras removes list when there is only ony feature map
            predictions = [predictions]
        predictions = {n: o for n, o in zip(keras_model.output_names, predictions)}

        for info in input_details:
            interpreter.set_tensor(info["index"], features[info["name"]].numpy())
        interpreter.invoke()
        tflite_predictions = {}
        for info in output_details:
            output = interpreter.get_tensor(info["index"])
            tflite_predictions[info["name"]] = output

        # match keras model predictions names to tflite (by name)
        keras_predictions = {}
        for k in predictions.keys():
            for lk in tflite_predictions.keys():
                if k in lk:
                    keras_predictions[lk] = predictions[k]
                    break

        assert len(keras_predictions) == len(tflite_predictions), (
            f"keras outputs: {keras_predictions.keys()}, "
            f"tflite outputs: {tflite_predictions.keys()}"
        )

        for n, ko in keras_predictions.items():
            lo = tflite_predictions[n]
            mae = np.abs(np.ravel(ko - lo)).mean()
            mse = (np.ravel(ko - lo) ** 2).mean()

            metrics[n]["mae"].append(mae)
            metrics[n]["rmse"].append(np.sqrt(mse))
            metrics[n]["keras_std"].append(ko.std())
            metrics[n]["keras_mean"].append(ko.mean())
            metrics[n]["tflite_std"].append(lo.std())
            metrics[n]["tflite_mean"].append(lo.mean())

    name_max_len = len(max(metrics.keys(), key=len))
    LOGGER.info("Measured deviation between keras and tflite model:")
    info_msg = ""
    for name in metrics.keys():
        mae = np.mean(metrics[name]["mae"])
        rmse = np.mean(metrics[name]["rmse"])
        keras_std = np.mean(metrics[name]["keras_std"])
        keras_mean = np.mean(metrics[name]["keras_mean"])
        tflite_std = np.mean(metrics[name]["tflite_std"])
        tflite_mean = np.mean(metrics[name]["tflite_mean"])
        name = name + " " * (1 + name_max_len - len(name))
        info_msg += (
            f"\n - {name} \n\tMAE     ={mae:10.6f} \n\tRMSE    ={rmse:10.6f} "
            f"\n\tKeras   = N(μ={keras_mean:10.6f}, σ={keras_std:10.6f})"
            f"\n\ttflite  = N(μ={tflite_mean:10.6f}, σ={tflite_std:10.6f})"
        )
    LOGGER.info(info_msg)
    return metrics
