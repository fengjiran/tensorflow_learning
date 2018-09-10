from __future__ import division
import os

import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Construct dcgan model."""

    def __init__(self, input_height=108, input_width=108, batch_size=16, sample_num=64):
        pass

    def generator(self, z, y=None):
        with tf.variable_scope('generator'):
            pass
