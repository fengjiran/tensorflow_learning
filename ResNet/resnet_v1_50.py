from __future__ import division
from __future__ import print_function

import tensorflow as tf

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d


class ResNet_v1_50(object):
    """Construct resnet v1 50."""

    def __init__(self, inputs, num_classes=1000, is_training=True, scope='resnet_v1_50'):
        self.inputs = inputs
        self.is_training = is_training
        self.num_classes = num_classes
