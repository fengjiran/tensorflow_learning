from __future__ import division
import os

import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Construct dcgan model."""

    def __init__(self):
        pass

    def generator(self, z, y=None):
        with tf.variable_scope('generator'):
            pass
