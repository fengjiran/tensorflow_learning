from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv


class InpaintingModel(object):
    """Construct model."""

    def __init__(self):
        print('Construct the inpainting model.')

    def edge_generator(self, x):
        pass

    def inpaint_generator(self, x):
        pass


def residual_block(x, in_channels, out_channels, dilation=1, name='residual_block'):
    pass


def instance_norm(x, name="instance_norm"):
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable("scale", [depth],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset
