from __future__ import print_function

import numpy as np
import tensorflow as tf


class Colorize(object):
    """Construct colorize model."""

    def __init__(self):
        print('constructing the model')
        self.conv_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.fc_init = tf.contrib.layers.xavier_initializer()
        self.activation = tf.nn.elu

    def low_level_network(self, inputs, reuse=None):
        cnum = 64
        with tf.variable_scope('shared_network', reuse=reuse):
            conv1 = self.activation(tf.layers.conv2d(inputs=inputs,
                                                     filters=cnum,
                                                     kernel_size=3,
                                                     strides=2,
                                                     padding='same',
                                                     name='conv1'))
            conv2 = self.activation(tf.layers.conv2d(inputs=conv1,
                                                     filters=2 * cnum,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding='same',
                                                     name='conv2'))

    def mid_level_network(self, inputs):
        pass

    def global_level_network(self, inputs):
        pass

    def colorize_network(self, inputs):
        pass
