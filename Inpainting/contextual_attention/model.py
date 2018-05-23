from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import spatial_discounting_mask
from utils import random_bbox
from utils import bbox2mask
from utils import local_patch
from utils import gan_wgan_loss
from utils import random_interpolates
from utils import gradient_penalty
from utils import images_summary
from utils import gradients_summary


class CompletionModel(object):
    """Construct model."""

    def __init__(self):
        pass

    def coarse_network(self, images, reuse=None):
        conv_layers = []
        cnum = 32

        with tf.variable_scope('coarse', reuse=reuse):
            conv1 = tf.layers.conv2d(images, cnum, 5, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv1')
            conv2 = tf.layers.conv2d(conv1, 2 * cnum, 3, strides=2, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv2_downsample')
            conv3 = tf.layers.conv2d(conv2, 2 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv3')
            conv4 = tf.layers.conv2d(conv3, 4 * cnum, 3, strides=2, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv4_downsample')
            conv5 = tf.layers.conv2d(conv4, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv5')
            conv6 = tf.layers.conv2d(conv5, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv6')

            conv7 = tf.layers.conv2d(conv6, 4 * cnum, 3, padding='same', dilation_rate=2, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv7_atrous')
            conv8 = tf.layers.conv2d(conv7, 4 * cnum, 3, padding='same', dilation_rate=4, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv8_atrous')
            conv9 = tf.layers.conv2d(conv8, 4 * cnum, 3, padding='same', dilation_rate=8, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv9_atrous')
            conv10 = tf.layers.conv2d(conv9, 4 * cnum, 3, padding='same', dilation_rate=16, activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv10_atrous')

            conv11 = tf.layers.conv2d(conv10, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv11')
            conv12 = tf.layers.conv2d(conv11, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv12')

            conv13 = tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv12,
                                                        (conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2])),
                filters=2 * cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.elu,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                name='conv13_upsample')

            conv14 = tf.layers.conv2d(conv13, 2 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv14')
            conv15 = tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv14,
                                                        (conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2])),
                filters=cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.elu,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                name='conv15_upsample')

            conv16 = tf.layers.conv2d(conv15, int(cnum / 2), 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv16')
            conv17 = tf.layers.conv2d(conv16, 3, 3, strides=1, padding='same',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv17')

            conv_output = tf.clip_by_value(conv17, -1., 1.)

            for i in range(1, 18):
                conv_layers.append(eval('conv{}'.format(i)))

            for conv in conv_layers:
                print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.get_shape().as_list()))

            return conv_output

    def refine_network(self, images, reuse=None):
        """Construct refine network."""
        conv_layers = []
        cnum = 32

        with tf.variable_scope('refine', reuse=reuse):
            conv1 = tf.layers.conv2d(images, cnum, 5, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv1')
            conv2 = tf.layers.conv2d(conv1, cnum, 3, strides=2, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv2_downsample')
            conv3 = tf.layers.conv2d(conv2, 2 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv3')
            conv4 = tf.layers.conv2d(conv3, 2 * cnum, 3, strides=2, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv4_downsample')
            conv5 = tf.layers.conv2d(conv4, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv5')
            conv6 = tf.layers.conv2d(conv5, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv6')

            conv7 = tf.layers.conv2d(conv6, 4 * cnum, 3, padding='same', dilation_rate=2, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv7_atrous')
            conv8 = tf.layers.conv2d(conv7, 4 * cnum, 3, padding='same', dilation_rate=4, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv8_atrous')
            conv9 = tf.layers.conv2d(conv8, 4 * cnum, 3, padding='same', dilation_rate=8, activation=tf.nn.elu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv9_atrous')
            conv10 = tf.layers.conv2d(conv9, 4 * cnum, 3, padding='same', dilation_rate=16, activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv10_atrous')

            conv11 = tf.layers.conv2d(conv10, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv11')
            conv12 = tf.layers.conv2d(conv11, 4 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv12')

            conv13 = tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv12,
                                                        (conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2])),
                filters=2 * cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.elu,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                name='conv13_upsample')
            conv14 = tf.layers.conv2d(conv13, 2 * cnum, 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv14')
            conv15 = tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv14,
                                                        (conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2])),
                filters=cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.elu,
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                name='conv15_upsample')
            conv16 = tf.layers.conv2d(conv15, int(cnum / 2), 3, strides=1, padding='same', activation=tf.nn.elu,
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv16')
            conv17 = tf.layers.conv2d(conv16, 3, 3, strides=1, padding='same',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv17')
            conv_output = tf.clip_by_value(conv17, -1., 1.)

            for i in range(1, 18):
                conv_layers.append(eval('conv{}'.format(i)))

            for conv in conv_layers:
                print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.get_shape().as_list()))

            return conv_output

    def global_discriminator(self, x, reuse=None):
        cnum = 64
        with tf.variable_scope('global_discriminator', reuse=reuse):
            conv1 = tf.layers.conv2d(x, cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv1')
            conv2 = tf.layers.conv2d(conv1, 2 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv2')
            conv3 = tf.layers.conv2d(conv2, 4 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv3')
            conv4 = tf.layers.conv2d(conv3, 4 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv4')

            return tf.contrib.layers.flatten(tf.nn.leaky_relu(conv4))

    def local_discriminator(self, x, reuse=None):
        cnum = 64
        with tf.variable_scope('local_discriminator', reuse=reuse):
            conv1 = tf.layers.conv2d(x, cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv1')
            conv2 = tf.layers.conv2d(conv1, 2 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv2')
            conv3 = tf.layers.conv2d(conv2, 4 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv3')
            conv4 = tf.layers.conv2d(conv3, 8 * cnum, 5, strides=2, padding='same', activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv4')

            return tf.contrib.layers.flatten(tf.nn.leaky_relu(conv4))

    def build_wgan_discriminator(self, global_input, local_input, reuse=None):
        with tf.variable_scope('wgan_discriminator', reuse=reuse):
            dglobal = self.global_discriminator(global_input, reuse=reuse)
            dlocal = self.local_discriminator(local_input, reuse=reuse)

            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')

            return dout_global, dout_local

    def build_graph_with_losses(self, batch_data, cfg, summary=True, reuse=None):
        batch_pos = batch_data / 127.5 - 1
        bbox = random_bbox(cfg)
        mask = bbox2mask(bbox, cfg)

        batch_incomplete = batch_pos * (1. - mask)
        ones_x = tf.ones_like(batch_incomplete)[:, :, :, 0:1]
        x = tf.concat([batch_incomplete, ones_x, ones_x * mask], axis=3)

        coarse_output = self.coarse_network(x, reuse)
        # apply mask and complete image
        batch_complete_coarse = coarse_output * mask + batch_incomplete * (1. - mask)
        refine_network_input = tf.concat([batch_complete_coarse, ones_x, ones_x * mask], axis=3)
        refine_output = self.refine_network(refine_network_input, reuse)

        if cfg['pretrain_coarse_network']:
            batch_predicted = coarse_output
        else:
            batch_predicted = refine_output

        losses = {}
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)

        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_coarse = local_patch(coarse_output, bbox)
        local_patch_refine = local_patch(refine_output, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)

        l1_alpha = cfg['coarse_l1_alpha']
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_coarse) *
                                                      spatial_discounting_mask(cfg))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - coarse_output) * (1. - mask))

        if not cfg['pretrain_coarse_network']:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_refine) *
                                                spatial_discounting_mask(cfg))
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - refine_output) * (1. - mask))
        losses['ae_loss'] /= tf.reduce_mean(1. - mask)

        if summary:
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])

            visual_img = [batch_pos, batch_incomplete, batch_complete]
            visual_img = tf.concat(visual_img, axis=2)
            images_summary(visual_img, 'raw_incomplete_predicted', 3)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        # local deteminator patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], axis=0)

        # wgan with gradient penalty
        pos_neg_global, pos_neg_local = self.build_wgan_discriminator(batch_pos_neg,
                                                                      local_patch_batch_pos_neg,
                                                                      reuse)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        pos_local, neg_local = tf.split(pos_neg_local, 2)

        # wgan loss
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global)
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local)
