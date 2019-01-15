import os
import numpy as np
import tensorflow as tf
from networks import InpaintingModel


class EdgeConnenct():
    """Construct edge connect model."""

    def __init__(self, config=None):
        self.cfg = config
