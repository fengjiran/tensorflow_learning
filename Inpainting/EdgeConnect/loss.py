import os
import inspect
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19(object):
    """Construct VGG19 model."""

    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            pass

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            pass
