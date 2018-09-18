from __future__ import division
import os

import numpy as np
import tensorflow as tf


class DCGAN(object):
    """Construct dcgan model."""

    def __init__(self):
        print('Construct the model.')
        self.g_vars = None
        self.d_vars = None

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

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        return outputs

    def discriminator(self, inputs, training, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.conv2d(inputs, 64, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
            x = tf.layers.batch_normalization(x, axis=list(range(len(x.get_shape()) - 1)), training=training)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 128, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
            x = tf.layers.batch_normalization(x, axis=list(range(len(x.get_shape()) - 1)), training=training)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 256, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
            x = tf.layers.batch_normalization(x, axis=list(range(len(x.get_shape()) - 1)), training=training)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 512, (5, 5), strides=(2, 2), padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
            x = tf.layers.batch_normalization(x, axis=list(range(len(x.get_shape()) - 1)), training=training)
            outputs = tf.nn.leaky_relu(x)

            batch_size = outputs.get_shape()[0].value
            outputs = tf.reshape(outputs, [batch_size, -1])
            outputs = tf.layers.dense(outputs, 2)

        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return outputs
