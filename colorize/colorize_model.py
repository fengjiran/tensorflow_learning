from __future__ import print_function

import numpy as np
import tensorflow as tf


class Colorize(object):
    """Construct colorize model."""

    def __init__(self):
        pass

    def low_level_network(self, inputs, reuse=None):
        cnum = 64
        with tf.variable_scope('shared_network', reuse=reuse):
            conv1 = tf.layers.conv2d(inputs, cnum, 3,
                                     strides=2,
                                     padding='same',
                                     name='conv1')

    def mid_level_network(self, inputs):
        pass

    def global_level_network(self, inputs):
        pass

    def colorize_network(self, inputs):
        pass
