from keras_detection.modules.core import Module
from typing import Optional
import tensorflow as tf
from keras_detection.modules.core import Module

keras = tf.keras
LOGGER = tf.get_logger()



class RPN(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
