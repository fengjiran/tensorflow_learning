from __future__ import division
from __future__ import print_function

import tensorflow as tf


# def trunc_normal(stddev): return tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
    """
    Define the Inception V1 base architecture.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to.
         can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
         'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
         'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
         'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: optional variable_scope

    Returns
    -------
    A dictionary from components of the network to the corresponding activation.

    """
    end_points = {}
    with tf.variable_scope(scope):
        end_point = 'Conv2d_1a_7x7'
        net = tf.layers.conv2d(inputs, 64, 7, strides=2, padding='same', activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                               name=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'MaxPool_2a_3x3'
        net = tf.layers.max_pooling2d(net, 3, strides=2, padding='same', name=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'Conv2d_2b_1x1'
        net = tf.layers.conv2d(net, 64, 1, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                               name=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'Conv2d_2c_3x3'
        net = tf.layers.conv2d(net, 192, 3, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                               name=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'MaxPool_3a_3x3'
        net = tf.layers.max_pooling2d(net, 3, strides=2, padding='same', name=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 96, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 128, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 32, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0b_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                            name='Conv2d_0a_1x1')
