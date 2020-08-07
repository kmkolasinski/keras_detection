from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List, Union, Any


import keras_detection.models.utils as kd_utils
from keras_detection.structures import ImageData
import tensorflow as tf


keras = tf.keras

ModuleOutput = Union[Any, Dict[str, Any]]
LossOutput = Union[tf.Tensor, Dict[str, tf.Tensor]]


class Module(ABC):
    def __call__(self, *args, **kwargs) -> ModuleOutput:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs) -> ModuleOutput:
        pass


class TrainableModule(keras.Model):
    def call(self, **kwargs) -> ModuleOutput:
        return super().call(**kwargs)


class NodeLoss(Module):
    def __init__(self, inputs: List[str], weight: float = 1):
        self.inputs = inputs
        self.weight = weight

    @abstractmethod
    def call(self, *args, **kwargs) -> LossOutput:
        pass


class InputNode:
    def __init__(self, name: str, getter):
        self.name = name
        self.getter = getter


class ImageInputNode(InputNode):
    def __init__(self):
        super().__init__(
            "image",
            getter=tf.autograph.experimental.do_not_convert(
                lambda d: ImageData.from_dict(d).features.image
            ),
        )


class LabelsInputNode(InputNode):
    def __init__(self):
        super().__init__(
            "labels",
            getter=tf.autograph.experimental.do_not_convert(
                lambda d: ImageData.from_dict(d).labels
            ),
        )


class Node:
    def __init__(
        self,
        name: str,
        inputs: List[str],
        module: Union[keras.Model, Any],
        loss: NodeLoss = None,
        inputs_as_list: bool = False,
        call_kwargs: Dict[str, Any] = {},
    ):
        self.name = name
        self.inputs = inputs
        self.module = module
        self.loss = loss
        self.inputs_as_list = inputs_as_list
        self.call_kwargs = call_kwargs


class NeuralGraph:
    def __init__(self):
        self.nodes = []
        self.input_nodes = []

    def add(self, node: Union[Node, InputNode]):
        if isinstance(node, Node):
            self.nodes.append(node)
        elif isinstance(node, InputNode):
            self.input_nodes.append(node)
        else:
            raise ValueError("Error node type", node)


class KerasGraph(keras.Model):
    def __new__(cls, graph: NeuralGraph, name):
        instance = super(KerasGraph, cls).__new__(cls, name=name)
        # make keras aware of all trainable layers
        instance.nodes = [n.module for n in graph.nodes]
        return instance

    def __init__(self, graph: NeuralGraph, name: str):
        super().__init__(name=name)
        self.graph = graph
        self.nodes_inputs_outputs = {}
        self.output_to_node = {}
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training: bool = False, mask=None):
        assert self.graph.input_nodes != []

        tensors = {}
        for node in self.graph.input_nodes:
            tensors[node.name] = node.getter(inputs)

        # TODO Fix me!
        tensors["image"] = tensors["image"] / 255.0

        for node in self.graph.nodes:
            inputs = [tensors[name] for name in node.inputs]
            print(f"> {node.name}({node.inputs}, inputs_as_list={node.inputs_as_list})")
            if node.inputs_as_list:
                outputs = node.module(inputs, training=training, **node.call_kwargs)
            else:
                outputs = node.module(*inputs, training=training, **node.call_kwargs)

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

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        **kwargs,
    ):
        super().compile(
            optimizer,
            loss,
            metrics,
            loss_weights,
            sample_weight_mode,
            weighted_metrics,
            **kwargs,
        )
        self.loss_tracker.reset_states()
        for node in self.graph.nodes:
            if node.loss is None:
                continue
            if isinstance(node.loss.loss_tracker, dict):
                for k, tracker in node.loss.loss_tracker.items():
                    tracker.reset_states()
            else:
                node.loss.loss_tracker.reset_states()

    def train_step(self, data):

        with tf.GradientTape() as tape:
            outputs = self(data, training=True)

            for k, v in self.nodes_inputs_outputs.items():
                if k not in outputs:
                    outputs[k] = v

            # TODO fix me!
            l2_loss = kd_utils.get_l2_loss_fn(l2_reg=1e-5, model=self)()
            total_loss = l2_loss

            nodes_losses = {}
            for node in self.graph.nodes:
                if node.loss is None:
                    continue
                inputs = [outputs[name] for name in node.loss.inputs]
                loss = node.loss.call(*inputs)
                if isinstance(loss, dict):
                    node_loss = {
                        k: node.loss.weight * tf.reduce_mean(l) for k, l in loss.items()
                    }
                    nodes_losses[node.name] = node_loss
                    total_loss = total_loss + tf.add_n(list(node_loss.values()))
                else:
                    node_loss = node.loss.weight * tf.reduce_mean(loss)
                    nodes_losses[node.name] = node_loss
                    total_loss = total_loss + node_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        metrics = {"loss": self.loss_tracker.result(), "l2_loss": l2_loss}

        for node in self.graph.nodes:
            if node.loss is None:
                continue
            loss = nodes_losses[node.name]
            if isinstance(loss, dict):
                for k, v in loss.items():
                    tracker = node.loss.loss_tracker[k]
                    tracker.update_state(v)
                    metrics[f"{node.name}/{tracker.name}"] = tracker.result()
            else:
                node.loss.loss_tracker.update_state(loss)
                tracker = node.loss.loss_tracker
                metrics[tracker.name] = tracker.result()
        return metrics
