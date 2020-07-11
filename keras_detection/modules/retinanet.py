import tensorflow as tf
from tensorflow.python.keras import Input

from keras_detection.modules.heads.heads import SingleConvHead
from keras_detection.targets.box_shape import BoxShapeTarget
from keras_detection.targets.box_objectness import BoxCenterIgnoreMarginObjectnessTarget, BoxCenterObjectnessTarget
from keras_detection.structures import FeatureMapDesc, ImageData
import keras_detection.losses as losses
from keras_detection.modules.backbones.resnet import ResNet
import keras_detection.models.utils as kd_utils

keras = tf.keras


class Retina(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        image_dim = 224
        self.backbone = ResNet(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=1,
        )

        self.box_head = SingleConvHead("box_shape", 4, activation=None)
        self.objectness_head = SingleConvHead("objectness", 1, activation="sigmoid")

        box_shape_ta = BoxShapeTarget()
        self.box_head.set_targets(box_shape_ta)
        self.box_head.set_losses(losses.L1Loss(box_shape_ta), 10.0)

        box_objectness_ta = BoxCenterIgnoreMarginObjectnessTarget(pos_weights=5.0)
        self.objectness_head.set_targets(box_objectness_ta)
        self.objectness_head.set_losses(
            losses.BCELoss(
                box_objectness_ta,
                label_smoothing=0.02,
                smooth_only_positives=True,
                from_logits=False,
            ),
            1.0,
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.box_loss_tracker = keras.metrics.Mean(name="box_loss")
        self.obj_loss_tracker = keras.metrics.Mean(name="obj_loss")
        self.l2_loss_tracker = keras.metrics.Mean(name="l2")

    def call(self, inputs, training: bool = False, mask=None):
        fm_id = 0
        if isinstance(inputs, dict):
            inputs = inputs['features']['image']

        image = inputs
        feature_maps = self.backbone(image)

        box_head_outputs = self.box_head(feature_maps[fm_id])
        objectness_head_outputs = self.objectness_head(feature_maps[fm_id])

        fm_desc = FeatureMapDesc(
            *feature_maps[fm_id].shape[1:3].as_list(), *image.shape[1:3].as_list()
        )

        self.box_head.set_feature_map_description(fm_desc)
        self.objectness_head.set_feature_map_description(fm_desc)

        return {"boxes": box_head_outputs, "objectness": objectness_head_outputs}

    def summary(self, line_length=None, positions=None, print_fn=None):
        image_dim = 224
        x = Input(shape=(image_dim, image_dim, 3))
        return keras.Model(inputs=x, outputs=self.call(x)).summary(
            line_length=line_length, positions=positions, print_fn=print_fn
        )

    def test_step(self, data):
        batch_data = ImageData.from_dict(data)

        predictions = self(batch_data.features.image / 255.0, training=True)

        box_loss_targets = self.box_head.compute_targets(batch_data)

        box_loss = self.box_head.compute_losses(
            box_loss_targets, predictions["boxes"]
        )

        obj_loss_targets = self.objectness_head.compute_targets(batch_data)
        obj_loss = self.objectness_head.compute_losses(
            obj_loss_targets, predictions["objectness"]
        )
        l2_reg_fn = kd_utils.get_l2_loss_fn(l2_reg=1e-5, model=self)()
        box_loss = tf.reduce_mean(box_loss)
        obj_loss = tf.reduce_mean(obj_loss)
        loss = box_loss + obj_loss + l2_reg_fn

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.box_loss_tracker.update_state(box_loss)
        self.obj_loss_tracker.update_state(obj_loss)
        self.l2_loss_tracker.update_state(l2_reg_fn)
        return {
            "loss": self.loss_tracker.result(),
            "box_loss": self.box_loss_tracker.result(),
            "obj_loss": self.obj_loss_tracker.result(),
            "l2_loss": self.l2_loss_tracker.result(),
        }

    def train_step(self, data):
        batch_data = ImageData.from_dict(data)

        with tf.GradientTape() as tape:
            predictions = self(batch_data.features.image / 255.0, training=True)

            box_loss_targets = self.box_head.compute_targets(batch_data)

            box_loss = self.box_head.compute_losses(
                box_loss_targets, predictions["boxes"]
            )

            obj_loss_targets = self.objectness_head.compute_targets(batch_data)
            obj_loss = self.objectness_head.compute_losses(
                obj_loss_targets, predictions["objectness"]
            )
            l2_reg_fn = kd_utils.get_l2_loss_fn(l2_reg=1e-5, model=self)()
            box_loss = tf.reduce_mean(box_loss)
            obj_loss = tf.reduce_mean(obj_loss)
            loss = box_loss + obj_loss + l2_reg_fn

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.box_loss_tracker.update_state(box_loss)
        self.obj_loss_tracker.update_state(obj_loss)
        self.l2_loss_tracker.update_state(l2_reg_fn)
        return {
            "loss": self.loss_tracker.result(),
            "box_loss": self.box_loss_tracker.result(),
            "obj_loss": self.obj_loss_tracker.result(),
            "l2_loss": self.l2_loss_tracker.result(),
        }
