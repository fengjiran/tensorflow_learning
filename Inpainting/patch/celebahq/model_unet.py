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
                                                           name='conv1')(images))
            conv2 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                           kernel_size=3,
                                                           strides=2,
                                                           padding='same',
                                                           name='conv2_downsample')(conv1))
            conv3 = self.activation(tf.keras.layers.Conv2D(filters=2 * cnum,
                                                           kernel_size=3,
                                                           strides=1,
                                                           padding='same',
                                                           name='conv3')(conv2))
            conv4 = self.activation(tf.keras.layers.Conv2D(filters=4 * cnum,
                                                           kernel_size=3,
                                                           strides=2,
                                                           padding='same',
                                                           name='conv4_downsample')(conv3))
