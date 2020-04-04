from typing import List

import tensorflow as tf
from keras_detection import FPNBuilder
from keras_detection.backbones import mobilenetv2_customized as mobilenetv2
from keras_detection.tasks import standard_tasks
import keras_detection.models.fpn_trainer as bd

_LOGGER = tf.get_logger()


class BoxCenterFPNTrainer(bd.FPNTrainer):
    def __init__(
        self,
        num_classes: int,
        image_dim: int = 128,
        mnet_alpha: float = 1.0,
        min_fm_size: int = 10,
        weights: str = "imagenet",
        label_smoothing: float = 0.01,
        box_shape_task: str = "box_shape",
        last_conv_filters: int = 64,
        class_activation: str = "sigmoid",
        box_obj_task: str = "center_ignore_margin",
    ):

        self._image_dim = image_dim
        self._num_classes = num_classes

        backbone = mobilenetv2.MobileNetV2Backbone(
            input_shape=(self._image_dim, self._image_dim, 3),
            alpha=mnet_alpha,
            min_fm_size=min_fm_size,
            weights=weights,
        )

        tasks = [
            standard_tasks.get_objectness_task(
                obj_class=box_obj_task,
                label_smoothing=label_smoothing,
                num_filters=last_conv_filters,
            ),
            standard_tasks.get_box_shape_task(
                box_shape_task, num_filters=last_conv_filters
            ),
            standard_tasks.get_multiclass_task(
                self._num_classes,
                fl_gamma=0.0,
                label_smoothing=label_smoothing,
                num_filters=last_conv_filters,
                activation=class_activation,
            ),
        ]
        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        super().__init__(builder)

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:

        warmup_length = 10
        base_lr = 0.01
        warmup_lr = 0.001
        scale_factor = 0.5

        def scheduler(epoch: int):
            if epoch < warmup_length:
                lr = warmup_lr + (base_lr - warmup_lr) * epoch / (warmup_length - 1)
            else:
                num_steps = epoch // warmup_length - 1
                lr = base_lr * (scale_factor ** num_steps)

            tf.summary.scalar("learning_rate", lr)
            _LOGGER.info(f"Using learning rate: {lr}")
            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        return [lr_callback]

    def get_optimizer(
        self, steps_per_epoch: int, run_mode: bd.RunMode, lr_scaling: int = 1
    ) -> tf.keras.optimizers.Optimizer:

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_2=0.99)
        return optimizer

    def build_model(self, batch_size: int = None, is_training=True) -> tf.keras.Model:
        return self.builder.build(batch_size=batch_size, is_training=is_training)


def get(ds: bd.Dataset) -> bd.FPNTrainer:
    return BoxCenterFPNTrainer(ds.num_classes)
