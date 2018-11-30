from __future__ import print_function

# import numpy as np
#
import tensorflow as tf

from utils import spatial_discounting_mask
from utils import random_bbox
from utils import bbox2mask
from utils import local_patch
from utils import gan_wgan_loss
from utils import random_interpolates
from utils import lipschitz_penalty
from utils import images_summary
from utils import gradients_summary


class CompletionModel(object):
    """Construct model."""

    def __init__(self):
        print('Construct the model')
        self.conv_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.fc_init = tf.contrib.layers.xavier_initializer()
        self.activation = tf.nn.elu
        # self.activation = tf.nn.leaky_relu
        self.norm_type = 'none'

    def coarse_network(self, images, reuse=None):
        conv_layers = []
        cnum = 32
        # if self.norm_type == 'instance_norm':
        #     norm = instance_norm
        # else:
        #     norm = tf.identity

        with tf.variable_scope('coarse', reuse=reuse):
            conv1 = self.activation(tf.keras.layers.Conv2D(filters=cnum,
                                                           kernel_size=5,
                                                           strides=1,
                                                           padding='same',
                                                           name='conv1')(images))  # 256, 256, 32
            conv2 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                           kernel_size=3,
                                                           strides=2,
                                                           padding='same',
                                                           name='conv2_downsample')(conv1))  # 128, 128, 64
            conv3 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           padding='same',
                                                           name='conv3')(conv2))  # 128, 128, 64
            conv4 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=2,
                                                           padding='same',
                                                           name='conv4_downsample')(conv3))  # 64, 64, 128
            conv5 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           padding='same',
                                                           name='conv5')(conv4))  # 64, 64, 128
            conv6 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           padding='same',
                                                           name='conv6')(conv5))  # 64, 64, 128
            conv7 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           dilation_rate=2,
                                                           padding='same',
                                                           name='conv7_atrous')(conv6))  # 64, 64, 128

            conv8_inputs = tf.concat([conv6, conv7], axis=-1)
            conv8 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           dilation_rate=4,
                                                           padding='same',
                                                           name='conv8_atrous')(conv8_inputs))  # 64, 64, 128
            conv9 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           dilation_rate=8,
                                                           padding='same',
                                                           name='conv9_atrous')(conv8))  # 64, 64, 128
            conv10 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            dilation_rate=16,
                                                            padding='same',
                                                            name='conv10_atrous')(conv9))  # 64, 64, 128
            conv11 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv11')(conv10))  # 64, 64, 128
            conv12 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv12')(conv11))  # 64, 64, 128

            conv13_inputs = tf.image.resize_nearest_neighbor(
                conv12,
                (conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2]))
            conv13_inputs = tf.concat([conv3, conv13_inputs], axis=-1)
            conv13 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv13_upsample')(conv13_inputs))  # 128, 128, 64
            conv14 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv14')(conv13))  # 128, 128, 64
            conv15_inputs = tf.image.resize_nearest_neighbor(
                conv14,
                (conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2])
            )

            conv15_inputs = tf.concat([conv1, conv15_inputs], axis=-1)
            conv15 = self.activation(tf.keras.layers.Conv2D(filters=cnum,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv15_upsampling')(conv15_inputs))  # 256, 256, 32
            conv16 = self.activation(tf.keras.layers.Conv2D(filters=cnum // 2,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv16')(conv15))  # 256, 256, 16
            conv17 = self.activation(tf.keras.layers.Conv2D(filters=3,
                                                            kernel_size=3,
                                                            strides=1,
                                                            padding='same',
                                                            name='conv17')(conv16))  # 256, 256, 3
            conv_output = tf.nn.tanh(conv17)

            for i in range(1, 18):
                conv_layers.append(eval('conv{}'.format(i)))

            for conv in conv_layers:
                print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.get_shape().as_list()))

            return conv_output


if __name__ == '__main__':
    model = CompletionModel()
    x = tf.random_uniform([10, 256, 256, 3])
    coarse = model.coarse_network(x)
