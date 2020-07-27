from typing import List, Union, Any, Dict
import keras_detection.losses as losses
from keras_detection import FeatureMapPredictionTarget
from keras_detection.losses import FeatureMapPredictionTargetLoss
from keras_detection.modular.backbones.core import FeatureMapDescEstimator
from keras_detection.modular.backbones.fpn import FPN
from keras_detection.modular.backbones.resnet import ResNet
from keras_detection.modular.core import (
    NodeLoss,
    NeuralGraph,
    Node,
    ImageInputNode,
    LabelsInputNode, TrainableModule,
)
from keras_detection.modular.heads.heads import SingleConvHead, Head
from keras_detection.structures import FeatureMapDesc, LabelsFrame
from keras_detection.targets.box_objectness import BoxCenterIgnoreMarginObjectnessTarget
from keras_detection.targets.box_shape import BoxShapeTarget

import tensorflow as tf

keras = tf.keras


class FeatureMapNodeLoss(NodeLoss):
    def __init__(
        self,
        inputs: List[str],
        name: str,
        target_assigner: FeatureMapPredictionTarget,
        loss: FeatureMapPredictionTargetLoss,
        weight: float = 1.0,
    ):
        super().__init__(inputs, weight=weight)
        self.box_shape_ta = target_assigner
        self.loss = loss
        self.loss_tracker = keras.metrics.Mean(name=name)
        self.fm_desc: FeatureMapDesc = None

    def call(
        self,
        fm_desc: FeatureMapDesc,
        batch_frame: LabelsFrame,
        predicted: tf.Tensor,
        per_anchor_loss: bool = False,
    ) -> tf.Tensor:
        targets = self.box_shape_ta.get_targets_tensors(fm_desc, batch_frame)
        self.fm_desc = fm_desc
        return self.loss.call(targets, predicted, per_anchor_loss=per_anchor_loss)

    def decode_predictions(self, outputs):
        return self.box_shape_ta.postprocess_predictions(self.fm_desc, outputs)


class BoxShapeLoss(FeatureMapNodeLoss):
    def __init__(self, inputs: List[str], weight: float = 5.0):
        box_shape_ta = BoxShapeTarget(use_tf_format=True)
        super().__init__(
            inputs,
            "box_shape",
            box_shape_ta,
            losses.L1Loss(box_shape_ta),
            weight=weight,
        )


class BoxObjectnessLoss(FeatureMapNodeLoss):
    def __init__(self, inputs: List[str], weight: float = 1.0):
        box_objectness_ta = BoxCenterIgnoreMarginObjectnessTarget(pos_weights=5.0)
        loss = losses.BCELoss(
            box_objectness_ta,
            label_smoothing=0.02,
            smooth_only_positives=True,
            from_logits=False,
        )
        super().__init__(inputs, "objectness", box_objectness_ta, loss, weight=weight)


class FeatureMapHead(TrainableModule):
    def __init__(self, name: str, head: Head, loss: FeatureMapNodeLoss, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.head = head
        self.head_loss = loss

    def call(self, feature_map: tf.Tensor, training: bool) -> Dict[str, Any]:
        predictions = self.head(feature_map, training=training)
        # TODO loss must have fm_desc defined
        postprocessed_outputs = self.head_loss.decode_predictions(predictions)
        outputs = {self.name: predictions, f"{self.name}/postprocessed": postprocessed_outputs}
        return outputs

    def as_node(self, inputs: List[str]) -> Node:
        return Node(
                self.name,
                inputs=inputs,
                module=self,
                loss=self.head_loss,
            )


class RetinaDetector(NeuralGraph):
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

        self.add(ImageInputNode())
        self.add(LabelsInputNode())

        self.add(Node("backbone", inputs=["image"], module=backbone))
        self.add(
            Node(
                "backbone/fm0/desc",
                inputs=["image", "backbone/fm0"],
                module=FeatureMapDescEstimator(),
            )
        )
        self.add(
            Node(
                "fpn",
                inputs=backbone.get_output_names("backbone"),
                inputs_as_list=True,
                module=fpn,
            )
        )

        self.add(
            FeatureMapHead(
                "boxes",
                head=SingleConvHead("boxes", 4, activation=None),
                loss=BoxShapeLoss(inputs=["backbone/fm0/desc", "labels", "boxes"]),
            ).as_node(inputs=["fpn/fm0"])
        )

        self.add(
            FeatureMapHead(
                "objectness",
                head=SingleConvHead("objectness", 1, activation="sigmoid"),
                loss=BoxObjectnessLoss(
                    inputs=["backbone/fm0/desc", "labels", "objectness"]
                ),
            ).as_node(inputs=["fpn/fm0"])
        )
