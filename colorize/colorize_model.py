from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.layers import conv2d
from tensorflow.layers import dense
from tensorflow.layers import flatten


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
            conv1 = self.activation(conv2d(inputs=inputs,
                                           filters=cnum,
                                           kernel_size=3,
                                           strides=2,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv1'))
            conv2 = self.activation(conv2d(inputs=conv1,
                                           filters=2 * cnum,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv2'))
            conv3 = self.activation(conv2d(inputs=conv2,
                                           filters=2 * cnum,
                                           kernel_size=3,
                                           strides=2,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv3'))
            conv4 = self.activation(conv2d(inputs=conv3,
                                           filters=4 * cnum,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv4'))
            conv5 = self.activation(conv2d(inputs=conv4,
                                           filters=4 * cnum,
                                           kernel_size=3,
                                           strides=2,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv5'))
            conv6 = self.activation(conv2d(inputs=conv5,
                                           filters=8 * cnum,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv6'))
            return conv6

    def mid_level_network(self, inputs, reuse=None):
        with tf.variable_scope('mid_level_network', reues=reuse):
            conv1 = self.activation(conv2d(inputs=inputs,
                                           filters=512,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv1'))
            conv2 = self.activation(conv2d(inputs=conv1,
                                           filters=256,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv2'))
            return conv2

    def global_level_network(self, inputs, reuse=None):
        with tf.variable_scope('global_level_network', reues=reuse):
            conv1 = self.activation(conv2d(inputs=inputs,
                                           filters=512,
                                           kernel_size=3,
                                           strides=2,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv1'))
            conv2 = self.activation(conv2d(inputs=conv1,
                                           filters=512,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv2'))
            conv3 = self.activation(conv2d(inputs=conv2,
                                           filters=512,
                                           kernel_size=3,
                                           strides=2,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv3'))
            conv4 = self.activation(conv2d(inputs=conv3,
                                           filters=512,
                                           kernel_size=3,
                                           strides=1,
                                           kernel_initializer=self.conv_init,
                                           padding='same',
                                           name='conv4'))
            flatted = flatten(conv4)
            fc1 = self.activation(dense(inputs=flatted,
                                        units=1024,
                                        kernel_initializer=self.fc_init,
                                        name='fc1'))
            fc2 = self.activation(dense(inputs=fc1,
                                        units=512,
                                        kernel_initializer=self.fc_init,
                                        name='fc2'))
            fc3 = self.activation(dense(inputs=fc2,
                                        units=256,
                                        kernel_initializer=self.fc_init,
                                        name='fc3'))
            return fc3

    def fusion(self, global_inputs, mid_inputs):
        with tf.variable_scope('fusion'):
            fusion_w = tf.get_variable(name='fusion_w',
                                       shape=[256, 512],
                                       initializer=self.fc_init,
                                       trainable=True)
            fusion_b = tf.get_variable('fusion_b',
                                       shape=[256],
                                       trainable=True)

    def colorize_network(self, inputs):
        pass
