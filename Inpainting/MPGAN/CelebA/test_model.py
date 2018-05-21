from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import FCLayer


def completion_network(images, batch_size):
    """Construct completion network."""
    # batch_size = images.get_shape().as_list()[0]
    # conv_layers = []
    cnum = 32
    input_channel = images.get_shape().as_list()[3]

    with tf.variable_scope('generator'):
        conv1 = Conv2dLayer(images, [5, 5, input_channel, cnum], stride=1, name='conv1')
        conv2 = Conv2dLayer(tf.nn.elu(conv1.output), [3, 3, cnum, 2 * cnum], stride=2, name='conv2_downsample')
        conv3 = Conv2dLayer(tf.nn.elu(conv2.output), [3, 3, 2 * cnum, 2 * cnum], stride=1, name='conv3')
        conv4 = Conv2dLayer(tf.nn.elu(conv3.output), [3, 3, 2 * cnum, 4 * cnum], stride=2, name='conv4_downsample')
        conv5 = Conv2dLayer(tf.nn.elu(conv4.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv5')
        conv6 = Conv2dLayer(tf.nn.elu(conv5.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv6')

        conv7 = DilatedConv2dLayer(tf.nn.elu(conv6.output), [3, 3, 4 * cnum, 4 * cnum], rate=2, name='conv7_atrous')
        conv8 = DilatedConv2dLayer(tf.nn.elu(conv7.output), [3, 3, 4 * cnum, 4 * cnum], rate=4, name='conv8_atrous')
        conv9 = DilatedConv2dLayer(tf.nn.elu(conv8.output), [3, 3, 4 * cnum, 4 * cnum], rate=8, name='conv9_atrous')
        conv10 = DilatedConv2dLayer(tf.nn.elu(conv9.output), [3, 3, 4 * cnum, 4 * cnum], rate=16, name='conv10_atrous')
        conv11 = Conv2dLayer(tf.nn.elu(conv10.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv11')
        conv12 = Conv2dLayer(tf.nn.elu(conv11.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv12')
        conv13 = DeconvLayer(inputs=tf.nn.elu(conv12.output),
                             filter_shape=[3, 3, 2 * cnum, 4 * cnum],
                             output_shape=[batch_size, conv4.output_shape[1],
                                           conv4.output_shape[2], 2 * cnum],
                             stride=1,
                             name='conv13_upsample')
        conv14 = Conv2dLayer(tf.nn.elu(conv13.output), [3, 3, 2 * cnum, 2 * cnum], stride=1, name='conv14')
        conv15 = DeconvLayer(inputs=tf.nn.elu(conv14.output),
                             filter_shape=[3, 3, cnum, 2 * cnum],
                             output_shape=[batch_size, conv2.output_shape[1],
                                           conv2.output_shape[2], 2 * cnum],
                             stride=1,
                             name='conv15_upsample')
        conv16 = Conv2dLayer(tf.nn.elu(conv15.output), [3, 3, cnum, int(cnum / 2)], stride=1, name='conv16')
        conv17 = Conv2dLayer(tf.nn.elu(conv16.output), [3, 3, int(cnum / 2), 3], stride=1, name='conv17')

        conv_output = tf.clip_by_value(conv17.output, -1., 1.)

        return conv_output
