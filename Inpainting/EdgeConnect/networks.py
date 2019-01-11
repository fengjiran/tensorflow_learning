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
        with tf.variable_scope('edge_generator'):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', name='conv3')
            x = instance_norm(x, name='in3')
            x = tf.nn.relu(x)

            return x

    def inpaint_generator(self, x):
        pass
