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
        with tf.variable_scope('mid_level_network', reuse=reuse):
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
        with tf.variable_scope('global_level_network', reuse=reuse):
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

            global_shape = global_inputs.get_shape().as_list()  # (N, 256)
            mid_shape = mid_inputs.get_shape().as_list()  # (N, h, w, 256)
            h = mid_shape[1]
            w = mid_shape[2]

            global_inputs = tf.reshape(global_inputs, [global_shape[0], 1, 1, global_shape[1]])  # (N, 1, 1, 256)
            global_inputs = tf.tile(global_inputs, [1, h, w, 1])  # (N, h, w, 256)

            fusion_inputs = tf.concat([global_inputs, mid_inputs], axis=-1)  # (N, h, w, 512)
            fusion_inputs = tf.reshape(fusion_inputs, [-1, 512])  # (Nhw, 512)
            fusion_output = tf.transpose(tf.matmul(fusion_w, tf.transpose(fusion_inputs))) + fusion_b
            # a = fusion_output.get_shape().as_list()
            # print(a)
            fusion_output = self.activation(tf.reshape(fusion_output, [mid_shape[0], h, w, 256]))
            return fusion_output

    def colorize_network(self, inputs):
        conv1 = self.activation(conv2d(inputs=inputs,
                                       filters=128,
                                       kernel_size=3,
                                       strides=1,
                                       kernel_initializer=self.conv_init,
                                       padding='same',
                                       name='conv1'))
        conv1_shape = conv1.get_shape().as_list()
        h = conv1_shape[1]
        w = conv1_shape[2]
        conv1_upsample = tf.image.resize_nearest_neighbor(conv1, [h * 2, w * 2])
        conv2 = self.activation(conv2d(inputs=conv1_upsample,
                                       filters=64,
                                       kernel_size=3,
                                       strides=1,
                                       kernel_initializer=self.conv_init,
                                       padding='same',
                                       name='conv2'))
        conv3 = self.activation(conv2d(inputs=conv2,
                                       filters=64,
                                       kernel_size=3,
                                       strides=1,
                                       kernel_initializer=self.conv_init,
                                       padding='same',
                                       name='conv3'))
        conv3_shape = conv3.get_shape().as_list()
        h = conv3_shape[1]
        w = conv3_shape[2]
        conv3_upsample = tf.image.resize_nearest_neighbor(conv3, [h * 2, w * 2])
        conv4 = self.activation(conv2d(inputs=conv3_upsample,
                                       filters=32,
                                       kernel_size=3,
                                       strides=1,
                                       kernel_initializer=self.conv_init,
                                       padding='same',
                                       name='conv4'))
        conv5 = tf.nn.sigmoid(conv2d(inputs=conv4,
                                     filters=2,
                                     kernel_size=3,
                                     strides=1,
                                     kernel_initializer=self.conv_init,
                                     padding='same',
                                     name='conv5'))
        return conv5


if __name__ == '__main__':
    model = Colorize()

    x = tf.random_uniform([10, 224, 224, 3])
    low_level_feature_for_mid = model.low_level_network(x)
    low_level_feature_for_global = model.low_level_network(x, reuse=True)

    mid_feature = model.mid_level_network(low_level_feature_for_mid)
    global_feature = model.global_level_network(low_level_feature_for_global)

    fusion = model.fusion(global_feature, mid_feature)
    output = model.colorize_network(fusion)
    print(output.get_shape().as_list())
