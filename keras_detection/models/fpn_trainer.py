import json
from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, List
import tensorflow as tf
from dataclasses import dataclass
from keras_detection import FPNBuilder
from importlib.machinery import SourceFileLoader
from tensorflow_estimator.python.estimator.export import export_lib
from keras_detection.datasets import prepare_dataset

_LOGGER = tf.get_logger()
_FINAL_MODEL_WEIGHTS = "weights-final.hdf5"
_FINAL_MODEL_HISTORY = "history-final.json"
keras = tf.keras


class RunMode(Enum):
    """
    Different training and evaluation modes:
        REGULAR - regular training i.e. starting from scratch
        TRANSFER - transfer weights from another model
        TRANSFER_AND_FREEZE - transfer weights from another model but keep them frozen during
            training
        FINETUNE - finalize training by finetuning whole network with specialized optimizer
            Finetune should be also used when running evaluation of the model in finetune
            mode.
    """

    REGULAR = "regular"
    TRANSFER = "transfer"
    TRANSFER_AND_FREEZE = "transfer_and_freeze"
    FINETUNE = "finetune"


class RunType(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


class Dataset:
    @property
    def num_classes(self) -> int:
        pass


@dataclass
class RunConfig:
    model_dir: str
    batch_size: int
    eval_steps: Optional[int] = None
    shuffle_buffer_size: Optional[int] = 512
    num_parallel_calls: Optional[int] = None
    prefetch_buffer_size: int = 2
    save_summary_steps: Optional[int] = None
    save_checkpoints_steps: Optional[int] = None
    log_count_steps: Optional[int] = None
    prefetch_to_device: Optional[str] = None

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.model_dir) / "checkpoints"

    @property
    def export_dir(self) -> Path:
        return Path(self.model_dir) / "export"

    @property
    def final_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / _FINAL_MODEL_WEIGHTS

    @property
    def final_checkpoint_history(self) -> Path:
        return self.checkpoints_dir / _FINAL_MODEL_HISTORY

    def load_final_checkpoint_history(self) -> tf.keras.callbacks.History:
        hist = tf.keras.callbacks.History()
        if self.final_checkpoint_history.exists():

            with self.final_checkpoint_history.open("r") as file:
                hist_params = json.load(file)

            hist.set_params(hist_params["params"])
            hist.epoch = hist_params["epoch"]
            hist.history = hist_params["history"]
            epoch = hist_params["params"]["epochs"]
            _LOGGER.info(
                f"Restoring latest checkpoint history. Starting training from epoch {epoch}"
            )
        else:
            hist.set_params({"epochs": 0, "steps": 0})
        return hist

    def save_final_checkpoint_history(self, hist: keras.callbacks.History):
        train_hist = {}
        for k, values in hist.history.items():
            train_hist[k] = [float(v) for v in values]
        hist_params = {
            "params": hist.params,
            "epoch": hist.epoch,
            "history": train_hist,
        }

        with self.final_checkpoint_history.open("w") as file:
            json.dump(hist_params, file)

    @property
    def tflite_export_path(self) -> Path:
        export_dir_base = str(self.export_dir)
        export_dir = export_lib.get_timestamped_export_dir(export_dir_base).decode()
        return Path(export_dir) / "model.tflite"


class FPNTrainer(metaclass=ABCMeta):
    """
    This class is used to build classification estimator used by
    train classifier script.
    """

    def __init__(self, builder: FPNBuilder):
        self._builder = builder
        self._train_callbacks = None
        # reference to the model during training
        self._model: Optional[keras.Model] = None

    @property
    def builder(self) -> FPNBuilder:
        if self._builder is None:
            raise ValueError("Model not initialized")
        return self._builder

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._builder.input_shape[:2]

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return []

    @abstractmethod
    def get_optimizer(
        self, steps_per_epoch: int, run_mode: RunMode, lr_scaling: int = 1
    ) -> tf.keras.optimizers.Optimizer:
        pass

    def get_build_training_targets_fn(self):
        return self.builder.get_build_training_targets_fn()

    def get_model_compile_args(self):
        return self.builder.get_model_compile_args()

    def build_model(
        self, batch_size: int = None, is_training: bool = True
    ) -> tf.keras.Model:
        return self.builder.build(batch_size=batch_size, is_training=is_training)

    @staticmethod
    def from_source_module(filepath: str, ds: Dataset) -> "FPNTrainer":
        model_module = SourceFileLoader("model_module", filepath).load_module()
        model_builder: FPNTrainer = model_module.get(ds=ds)
        return model_builder

    def prepare_dataset(
        self, dataset: tf.data.Dataset, config: RunConfig, num_epochs: int = -1
    ) -> tf.data.Dataset:

        train_dataset = prepare_dataset(
            dataset=dataset,
            model_image_size=self.input_shape,
            batch_size=config.batch_size,
            num_epochs=num_epochs,
            shuffle_buffer_size=config.shuffle_buffer_size,
            prefetch_buffer_size=config.prefetch_buffer_size,
            num_parallel_calls=config.num_parallel_calls,
            prefetch_to_device=config.prefetch_to_device,
        )

        return train_dataset.map(self.get_build_training_targets_fn())

    def restore(
        self, checkpoint: Path, batch_size: int = None, is_training: bool = True
    ) -> tf.keras.Model:
        model = self.build_model(batch_size=batch_size, is_training=is_training)
        restore_path = Path(checkpoint)
        if restore_path.exists():
            _LOGGER.info(f"Restoring weights: {restore_path}")
            model.load_weights(str(restore_path))
        else:
            _LOGGER.info(f"Cannot restore model weights: {restore_path}")

        return model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        num_epochs: int,
        steps_per_epoch: int,
        config: RunConfig,
        eval_dataset: Optional[tf.data.Dataset] = None,
        tensorboard: bool = True,
    ) -> keras.Model:

        model = self.build_model(batch_size=config.batch_size, is_training=True)

        restore_path = config.final_checkpoint_path
        if not restore_path.parent.exists():
            restore_path.parent.mkdir(parents=True)

        if restore_path.exists():
            _LOGGER.info(f"Restoring weights: {restore_path}")
            model.load_weights(str(restore_path))

        prepared_train_dataset = self.prepare_dataset(
            dataset=train_dataset, config=config,
        )

        optimizer = self.get_optimizer(
            steps_per_epoch=steps_per_epoch, run_mode=RunMode.REGULAR
        )
        model.compile(optimizer, **self.get_model_compile_args())
        self._model = model
        _LOGGER.info(f"Monitored metrics: {model.metrics_names}")

        history_callback = config.load_final_checkpoint_history()
        initial_epoch = history_callback.params["epochs"]
        callbacks = [history_callback]

        if tensorboard:
            tensorboard_callback = keras.callbacks.TensorBoard(
                Path(config.model_dir) / "logs",
                profile_batch=0,
                update_freq=config.save_summary_steps or steps_per_epoch,
            )
            callbacks.append(tensorboard_callback)

        validation_data = None
        validation_steps = None
        if eval_dataset is not None:
            if config.eval_steps is None:
                raise ValueError(
                    "Eval steps cannot be none when using evaluation dataset"
                )
            validation_steps = config.eval_steps
            if not config.checkpoints_dir.exists():
                config.checkpoints_dir.mkdir(parents=True)

            filepath = (
                config.checkpoints_dir / "weights.{epoch:04d}-{val_loss:.3f}.hdf5"
            )
            ckpt_callback = keras.callbacks.ModelCheckpoint(
                filepath=str(filepath),
                monitor="val_loss",
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode="min",
                save_freq="epoch",
            )
            validation_data = self.prepare_dataset(
                dataset=train_dataset, config=config,
            )
            callbacks.append(ckpt_callback)

        callbacks = callbacks + self.get_callbacks()
        self._train_callbacks = callbacks
        model.fit(
            prepared_train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            initial_epoch=initial_epoch,
        )

        model.save_weights(str(restore_path))
        config.save_final_checkpoint_history(model.history)
        return model

    def export_to_tflite(
        self,
        test_dataset: tf.data.Dataset,
        config: RunConfig,
        export_batch_size: int = 1,
        num_dataset_samples: int = 64,
        num_test_steps: int = 16,
        postprocess_outputs: bool = True,
        merge_feature_maps: bool = True,
        test_converted_models: bool = True,
        convert_quantized_model: bool = True,
    ) -> Tuple[keras.Model, List[Path]]:

        exported_model, tflite_models_paths = self.builder.convert_to_tflite(
            config.final_checkpoint_path,
            save_path=str(config.tflite_export_path),
            export_batch_size=export_batch_size,
            raw_dataset=test_dataset,
            num_dataset_samples=num_dataset_samples,
            num_test_steps=num_test_steps,
            merge_feature_maps=merge_feature_maps,
            postprocess_outputs=postprocess_outputs,
            convert_quantized_model=convert_quantized_model,
            verify_converted_models=test_converted_models,
        )

        return exported_model, tflite_models_paths
