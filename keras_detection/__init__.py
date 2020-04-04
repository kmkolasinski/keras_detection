import logging

from keras_detection.backbones.base import Backbone
from keras_detection.targets.base import FeatureMapPredictionTarget, FeatureMapDesc
from keras_detection.structures import LabelsFrame, ImageData, Features
from keras_detection.models.fpn_builder import FPNBuilder

# keep single stream handler
from tensorflow.python.platform.tf_logging import _logger

if len(_logger.handlers) > 1:
    stream_handlers = []
    other_handlers = []
    for handler in _logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handlers.append(handler)
        else:
            other_handlers.append(handler)

    if len(stream_handlers) > 1:
        stream_handlers = stream_handlers[:1]
    _logger.handlers = stream_handlers + other_handlers
