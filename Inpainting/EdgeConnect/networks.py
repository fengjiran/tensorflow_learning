from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm


class InpaintingModel(object):
    """Construct model."""

    def __init__(self):
        print('Construct the inpainting model.')

    def edge_generator(self, x):
        pass

    def inpaint_generator(self, x):
        pass
