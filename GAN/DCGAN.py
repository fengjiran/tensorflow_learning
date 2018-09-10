from __future__ import division
import os

import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Construct dcgan model."""

    def __init__(self, input_height=108, input_width=108, batch_size=16, sample_num=64):
        print('Construct the model.')

    def generator(self, z, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = tf.layers.dense(z, 256 * 8 * 8, activation=tf.nn.relu)
