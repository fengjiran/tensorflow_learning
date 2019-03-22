from __future__ import print_function
import os
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss

from utils import images_summary


class EdgeModel():
    """Construct edge model."""

    def __init__(self, config=None):
        print('Construct the edge model.')
        self.cfg = config
        self.init_type = self.cfg['INIT_TYPE']

    def edge_generator(self, x, reuse=None):
        with tf.variable_scope('edge_generator', reuse=reuse):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv1')
