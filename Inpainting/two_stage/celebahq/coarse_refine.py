import os
import numpy as np
import tensorflow as tf
from .networks import InpaintingModel
from.dataset import Dataset


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)
        self.train_dataset = Dataset(config)

    def train(self):
        pass
