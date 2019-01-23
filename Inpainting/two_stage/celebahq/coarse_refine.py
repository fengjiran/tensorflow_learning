import os
import numpy as np
import tensorflow as tf
from .networks import InpaintingModel


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)

    def train(self):
        pass
