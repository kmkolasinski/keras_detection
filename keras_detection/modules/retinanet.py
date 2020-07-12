from abc import abstractmethod
from typing import List, Union, Any

import tensorflow as tf
from tensorflow.python.keras import Input

from keras_detection.modules.backbones.fpn import FPN
from keras_detection.modules.heads.heads import SingleConvHead
from keras_detection.targets.box_shape import BoxShapeTarget
from keras_detection.targets.box_objectness import (
    BoxCenterIgnoreMarginObjectnessTarget,
    BoxCenterObjectnessTarget,
)
from keras_detection.structures import FeatureMapDesc, ImageData, LabelsFrame
import keras_detection.losses as losses
from keras_detection.modules.backbones.resnet import ResNet, FeatureMapDescEstimator
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
            inputs = inputs["features"]["image"]

        image = inputs / 255.0
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

        predictions = self(batch_data.features.image, training=True)

        box_loss_targets = self.box_head.compute_targets(batch_data)

        box_loss = self.box_head.compute_losses(box_loss_targets, predictions["boxes"])

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
            predictions = self(batch_data.features.image, training=True)

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


class NodeLoss:
    def __init__(self, inputs: List[str], weight: float = 1):
        self.inputs = inputs
        self.weight = weight

    @abstractmethod
    def call(self, *args, **kwargs) -> tf.Tensor:
        pass


class BoxShapeLoss(NodeLoss):
    def __init__(self, inputs: List[str]):
        super().__init__(inputs, weight=10.0)
        self.box_shape_ta = BoxShapeTarget()
        self.loss = losses.L1Loss(self.box_shape_ta)
        self.loss_tracker = keras.metrics.Mean(name="box_loss")
        self.fm_desc: FeatureMapDesc = None

    def call(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame, predicted: tf.Tensor
    ) -> tf.Tensor:
        targets = self.box_shape_ta.get_targets_tensors(fm_desc, batch_frame)
        self.fm_desc = fm_desc
        return self.loss.call(targets, predicted)

    def decode_predictions(self, outputs):
        return self.box_shape_ta.postprocess_predictions(self.fm_desc, outputs)


class BoxObjectnessLoss(NodeLoss):
    def __init__(self, inputs: List[str]):
        super().__init__(inputs, weight=1.0)
        self.box_objectness_ta = BoxCenterIgnoreMarginObjectnessTarget(pos_weights=5.0)
        self.loss = losses.BCELoss(
            self.box_objectness_ta,
            label_smoothing=0.02,
            smooth_only_positives=True,
            from_logits=False,
        )
        self.loss_tracker = keras.metrics.Mean(name="objectness")
        self.fm_desc: FeatureMapDesc = None

    def call(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame, predicted: tf.Tensor
    ) -> tf.Tensor:
        targets = self.box_objectness_ta.get_targets_tensors(fm_desc, batch_frame)
        self.fm_desc = fm_desc
        return self.loss.call(targets, predicted)

    def decode_predictions(self, outputs):
        return self.box_objectness_ta.postprocess_predictions(self.fm_desc, outputs)


class Node:
    def __init__(
        self,
        name: str,
        inputs: List[str],
        net: Union[keras.Model, Any],
        loss: NodeLoss = None,
        inputs_as_list: bool = False
    ):
        self.name = name
        self.inputs = inputs
        self.net = net
        self.loss = loss
        self.inputs_as_list = inputs_as_list


class NeuralGraph:
    def __init__(self):
        self.nodes = []

    def add(self, node: Node):
        self.nodes.append(node)


class KerasGraph(keras.Model):
    def __new__(cls, graph, name):
        instance = super(KerasGraph, cls).__new__(cls, name=name)
        # make keras aware of all trainable layers
        instance.nodes = [n.net for n in graph.nodes]
        return instance

    def __init__(self, graph, name: str):
        super().__init__(name=name)
        self.graph = graph
        self.nodes_inputs_outputs = {}
        self.output_to_node = {}
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training: bool = False, mask=None):
        image = inputs
        tensors = {"image": image / 255.0}
        for node in self.graph.nodes:
            inputs = [tensors[name] for name in node.inputs]

            if node.inputs_as_list:
                outputs = node.net(inputs, training=training)
            else:
                outputs = node.net(*inputs, training=training)

            node_name = node.name
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    tensors[f"{node_name}/{k}"] = v
                    self.output_to_node[f"{node_name}/{k}"] = node
            else:
                tensors[node_name] = outputs
                self.output_to_node[node_name] = node

        self.nodes_inputs_outputs = tensors
        return {k: v for k, v in tensors.items() if isinstance(v, tf.Tensor)}

    def train_step(self, data):
        batch_data = ImageData.from_dict(data)

        with tf.GradientTape() as tape:
            outputs = self(batch_data.features.image, training=True)
            outputs["labels"] = batch_data.labels
            for k, v in self.nodes_inputs_outputs.items():
                if k not in outputs:
                    outputs[k] = v

            nodes_losses = {}
            for node in self.graph.nodes:
                if node.loss is None:
                    continue
                inputs = [outputs[name] for name in node.loss.inputs]

                loss = node.loss.call(*inputs)
                nodes_losses[node.name] = node.loss.weight * tf.reduce_mean(loss)

            l2_loss = kd_utils.get_l2_loss_fn(l2_reg=1e-5, model=self)()
            total_loss = l2_loss
            for k, v in nodes_losses.items():
                total_loss = total_loss + v

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        metrics = {}
        for node in self.graph.nodes:
            if node.loss is None:
                continue
            node.loss.loss_tracker.update_state(nodes_losses[node.name])
            tracker = node.loss.loss_tracker
            metrics[tracker.name] = tracker.result()

        self.loss_tracker.update_state(total_loss)
        metrics["loss"] = self.loss_tracker.result()
        return metrics


class RetinaGraph(NeuralGraph):
    def __init__(self):
        super().__init__()

        image_dim = 224
        backbone = ResNet(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1, 1),
            num_last_blocks=2,
        )

        fpn = FPN(
            input_shapes=backbone.get_output_shapes(), depth=64, num_first_blocks=1
        )

        box_head = SingleConvHead("box_shape", 4, activation=None)
        objectness_head = SingleConvHead("objectness", 1, activation="sigmoid")

        self.add(Node("backbone", inputs=["image"], net=backbone))
        self.add(
            Node(
                "backbone/fm0/desc",
                inputs=["image", "backbone/fm0"],
                net=FeatureMapDescEstimator(),
            )
        )
        self.add(
            Node(
                "fpn",
                inputs=backbone.get_output_names("backbone"),
                inputs_as_list=True,
                net=fpn,
            )
        )
        self.add(
            Node(
                "boxes",
                inputs=["fpn/fm0"],
                net=box_head,
                loss=BoxShapeLoss(inputs=["backbone/fm0/desc", "labels", "boxes"]),
            )
        )
        self.add(
            Node(
                "objectness",
                inputs=["fpn/fm0"],
                net=objectness_head,
                loss=BoxObjectnessLoss(
                    inputs=["backbone/fm0/desc", "labels", "objectness"]
                ),
            )
        )
