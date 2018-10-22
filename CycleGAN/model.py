from __future__ import division
import tensorflow as tf
from ops import batch_norm
from ops import instance_norm


class CycleGAN(object):
    """Build cyclegan model."""

    def __init__(self):
        pass

    def discriminator(self, image, options, reuse=None, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            pass
