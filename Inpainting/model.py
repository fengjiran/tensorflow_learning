from __future__ import print_function

import numpy as np
import pickle
import tensorflow as tf


def conv_layer(inputs, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=filter_shape[-1],
                            initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(inputs, w, [1, stride, stride, 1], padding=padding)

    return activation(tf.nn.bias_add(conv, b))


def deconv_layer(inputs, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=filter_shape[-2],
                            initializer=tf.constant_initializer(0.))

        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)
    return activation(tf.nn.bias_add(deconv, b))


def fc_layer(inputs, output_size, activation=tf.identity, name=None):
    shape = inputs.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(inputs, [-1, dim])   # flatten
    input_size = dim

    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[output_size],
                            initializer=tf.constant_initializer(0.))

    return activation(tf.nn.bias_add(tf.matmul(x, w), b))
