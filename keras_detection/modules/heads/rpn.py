from keras_detection import FeatureMapDesc
from keras_detection.losses import BCELoss
from keras_detection.modules.core import Module
from typing import Optional, List, Tuple
import tensorflow as tf
from keras_detection.modules.core import Module
from keras_detection.modules.heads.heads import SingleConvHead
from keras_detection.targets.box_objectness import BoxCenterIgnoreMarginObjectnessTarget
from keras_detection.targets.box_shape import BoxShapeTarget

keras = tf.keras
LOGGER = tf.get_logger()


class RPN(Module):

    def __init__(self, name: str = "RPN", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.box_head = SingleConvHead("rpn/box_shape", 4, activation=None)
        self.objectness_head = SingleConvHead("rpn/objectness", 1, activation="sigmoid")
        self.box_shape_ta = BoxShapeTarget()

    def call(self, fm_desc: FeatureMapDesc, feature_map: tf.Tensor, training: bool = None, mask=None):
        raw_boxes = self.box_head(feature_map)
        objectness = self.objectness_head(feature_map)
        proposals = self.box_shape_ta.to_tf_boxes(self.box_shape_ta.postprocess_predictions(fm_desc, raw_boxes))
        return {"proposals": proposals, "raw_boxes": raw_boxes, "objectness": objectness}
