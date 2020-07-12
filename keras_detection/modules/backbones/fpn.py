from typing import Tuple, List
from keras_detection.backbones.fpn import build_fpn
from keras_detection.modules.core import Module
import tensorflow as tf


class FPN(Module):
    def __init__(
        self,
        input_shapes: List[Tuple[int, int, int]],
        depth: int = 64,
        num_first_blocks: int = 1,
        name: str = "FPN",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.num_first_blocks = num_first_blocks
        self.fpn = build_fpn(input_shapes, depth, name=f"{name}/model")

    def call(self, inputs, training=None, mask=None):
        outputs = self.fpn(inputs)
        if isinstance(outputs, tf.Tensor):
            outputs = [outputs]
        outputs = outputs[: self.num_first_blocks]
        named_outputs = {}
        for fm_id, o in enumerate(outputs):
            named_outputs[f"fm{fm_id}"] = o
        return named_outputs
