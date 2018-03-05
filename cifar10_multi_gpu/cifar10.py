"""Builds the CIFAR-10 network."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# batch_size=128


def _variable_on_cpu(name, shape, initializer):
    """Help to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns
    -------
        Variable Tensor

    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    return var


class Conv2dLayer(object):
    """Construct conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 b_init=0.,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = _variable_on_cpu(name='w',
                                      shape=filter_shape,
                                      initializer=tf.glorot_normal_initializer())

            self.b = _variable_on_cpu(name='b',
                                      shape=filter_shape[-1],
                                      initializer=tf.constant_initializer(b_init))

            linear_output = tf.nn.conv2d(self.inputs, self.w, [1, stride, stride, 1], padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


def inference(images, batch_size=128):

    with tf.variable_scope('inference'):
        # conv1
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], tf.nn.relu, name='conv1')
        # pool1
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        conv2 = Conv2dLayer(norm1, [5, 5, 64, 64], tf.nn.relu, b_init=0.1, name='conv1')
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_on_cpu(name='weights',
                                   shape=[dim, 384],
                                   initializer=tf.glorot_normal_initializer())

        biases = _variable_on_cpu(name='biases',
                                  shape=[384],
                                  initializer=tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
