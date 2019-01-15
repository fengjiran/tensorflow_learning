import os
import numpy as np
import tensorflow as tf
from networks import InpaintingModel


class EdgeConnenct():
    """Construct edge connect model."""

    def __init__(self, config=None):
        self.cfg = config

        if self.cfg['MODEL'] == 1:
            model_name = 'edge'
        elif self.cfg['MODEL'] == 2:
            model_name = 'inpaint'
        elif self.cfg['MODEL'] == 3:
            model_name = 'edge_inpaint'
        elif self.cfg['MODEL'] == 4:
            model_name = 'joint'

    def train(self):
        epoch = 0
        keep_training = True
        model = self.cfg['MODEL']
        max_iteration = int(self.cfg['MAX_ITERS'])
