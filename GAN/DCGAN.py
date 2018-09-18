from __future__ import division
import os

import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Construct dcgan model."""

    def __init__(self):
        print('Construct the model.')

    def generator(self, z, training, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            x = tf.layers.dense(z, 1024 * 4 * 4,
                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                name='layer1')
            x = tf.reshape(x, (-1, 4, 4, 1024))
            x = tf.layers.batch_normalization(x, axis=list(
                range(len(x.get_shape()) - 1)), training=training, name='bn1')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, 512, (5, 5), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           name='layer2')
            x = tf.layers.batch_normalization(x, axis=list(
                range(len(x.get_shape()) - 1)), training=training, name='bn2')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, 256, (5, 5), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           name='layer3')
            x = tf.layers.batch_normalization(x, axis=list(
                range(len(x.get_shape()) - 1)), training=training, name='bn3')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, 128, (5, 5), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           name='layer4')
            x = tf.layers.batch_normalization(x, axis=list(
                range(len(x.get_shape()) - 1)), training=training, name='bn4')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d_transpose(x, 3, (5, 5), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                           name='layer5')
            outputs = tf.tanh(x, name='outputs')

        return outputs
