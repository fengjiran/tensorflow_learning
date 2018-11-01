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
# from utils import instance_norm


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
            conv1 = self.activation(tf.layers.conv2d(images, cnum, 5,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv1'))
            conv2 = self.activation(tf.layers.conv2d(conv1, 2 * cnum, 3,
                                                     strides=2,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv2_downsample'))
            conv3 = self.activation(tf.layers.conv2d(conv2, 2 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv3'))
            conv4 = self.activation(tf.layers.conv2d(conv3, 4 * cnum, 3,
                                                     strides=2,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv4_downsample'))
            conv5 = self.activation(tf.layers.conv2d(conv4, 4 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv5'))
            conv6 = self.activation(tf.layers.conv2d(conv5, 4 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv6'))

            conv7 = self.activation(tf.layers.conv2d(conv6, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=2,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv7_atrous'))
            conv8 = self.activation(tf.layers.conv2d(conv7, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=4,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv8_atrous'))
            conv9 = self.activation(tf.layers.conv2d(conv8, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=8,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv9_atrous'))
            conv10 = self.activation(tf.layers.conv2d(conv9, 4 * cnum, 3,
                                                      padding='same',
                                                      dilation_rate=16,
                                                      kernel_initializer=self.conv_init,
                                                      name='conv10_atrous'))

            conv11 = self.activation(tf.layers.conv2d(conv10, 4 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv11'))
            conv12 = self.activation(tf.layers.conv2d(conv11, 4 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv12'))

            conv13 = self.activation(tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv12,
                                                        (conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2])),
                filters=2 * cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=self.conv_init,
                name='conv13_upsample'))

            conv14 = self.activation(tf.layers.conv2d(conv13, 2 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv14'))
            conv15 = self.activation(tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv14,
                                                        (conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2])),
                filters=cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=self.conv_init,
                name='conv15_upsample'))

            conv16 = self.activation(tf.layers.conv2d(conv15, cnum // 2, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv16'))
            conv17 = tf.layers.conv2d(conv16, 3, 3,
                                      strides=1,
                                      padding='same',
                                      kernel_initializer=self.conv_init,
                                      name='conv17')

            # conv_output = tf.clip_by_value(conv17, -1., 1.)
            conv_output = tf.nn.tanh(conv17)

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
            conv1 = self.activation(tf.layers.conv2d(images, cnum, 5,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv1'))
            conv2 = self.activation(tf.layers.conv2d(conv1, cnum, 3,
                                                     strides=2,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv2_downsample'))
            conv3 = self.activation(tf.layers.conv2d(conv2, 2 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv3'))
            conv4 = self.activation(tf.layers.conv2d(conv3, 2 * cnum, 3,
                                                     strides=2,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv4_downsample'))
            conv5 = self.activation(tf.layers.conv2d(conv4, 4 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv5'))
            conv6 = self.activation(tf.layers.conv2d(conv5, 4 * cnum, 3,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=self.conv_init,
                                                     name='conv6'))

            conv7 = self.activation(tf.layers.conv2d(conv6, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=2,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv7_atrous'))
            conv8 = self.activation(tf.layers.conv2d(conv7, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=4,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv8_atrous'))
            conv9 = self.activation(tf.layers.conv2d(conv8, 4 * cnum, 3,
                                                     padding='same',
                                                     dilation_rate=8,
                                                     kernel_initializer=self.conv_init,
                                                     name='conv9_atrous'))
            conv10 = self.activation(tf.layers.conv2d(conv9, 4 * cnum, 3,
                                                      padding='same',
                                                      dilation_rate=16,
                                                      kernel_initializer=self.conv_init,
                                                      name='conv10_atrous'))

            conv11 = self.activation(tf.layers.conv2d(conv10, 4 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv11'))
            conv12 = self.activation(tf.layers.conv2d(conv11, 4 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv12'))

            conv13 = self.activation(tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv12,
                                                        (conv3.get_shape().as_list()[1], conv3.get_shape().as_list()[2])),
                filters=2 * cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=self.conv_init,
                name='conv13_upsample'))
            conv14 = self.activation(tf.layers.conv2d(conv13, 2 * cnum, 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv14'))
            conv15 = self.activation(tf.layers.conv2d(
                inputs=tf.image.resize_nearest_neighbor(conv14,
                                                        (conv1.get_shape().as_list()[1], conv1.get_shape().as_list()[2])),
                filters=cnum,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=self.conv_init,
                name='conv15_upsample'))
            conv16 = self.activation(tf.layers.conv2d(conv15, int(cnum / 2), 3,
                                                      strides=1,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv16'))
            conv17 = tf.layers.conv2d(conv16, 3, 3,
                                      strides=1,
                                      padding='same',
                                      kernel_initializer=self.conv_init,
                                      name='conv17')
            # conv_output = tf.clip_by_value(conv17, -1., 1.)
            conv_output = tf.nn.tanh(conv17)

            for i in range(1, 18):
                conv_layers.append(eval('conv{}'.format(i)))

            for conv in conv_layers:
                print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.get_shape().as_list()))

            return conv_output

    def global_discriminator(self, x, reuse=None):
        cnum = 64
        with tf.variable_scope('global_discriminator', reuse=reuse):
            conv1 = tf.nn.leaky_relu(tf.layers.conv2d(x, cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv1'))
            conv2 = tf.nn.leaky_relu(tf.layers.conv2d(conv1, 2 * cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv2'))
            conv3 = tf.nn.leaky_relu(tf.layers.conv2d(conv2, 4 * cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv3'))
            conv4 = tf.nn.leaky_relu(tf.layers.conv2d(conv3, 4 * cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv4'))

            return tf.contrib.layers.flatten(conv4)

    def local_discriminator(self, x, reuse=None):
        cnum = 64
        with tf.variable_scope('local_discriminator', reuse=reuse):
            conv1 = tf.nn.leaky_relu(tf.layers.conv2d(x, cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv1'))
            conv2 = tf.nn.leaky_relu(tf.layers.conv2d(conv1, 2 * cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv2'))
            conv3 = tf.nn.leaky_relu(tf.layers.conv2d(conv2, 4 * cnum, 5,
                                                      strides=2,
                                                      padding='same',
                                                      kernel_initializer=self.conv_init,
                                                      name='conv3'))
            conv4 = tf.layers.conv2d(conv3, 4 * cnum, 5,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=self.conv_init,
                                     name='conv4')
            # conv5 = tf.layers.conv2d(conv4, 1, 1, padding='same', activation=tf.nn.leaky_relu,
            #                          kernel_initializer=tf.keras.initializers.glorot_normal(), name='conv5')
            return tf.reduce_mean(conv4, axis=[1, 2, 3])  # tf.contrib.layers.flatten(tf.nn.leaky_relu(conv5))

    def build_wgan_discriminator(self, global_input, local_input, reuse=None):
        with tf.variable_scope('wgan_discriminator', reuse=reuse):
            dglobal = self.global_discriminator(global_input, reuse=reuse)
            dlocal = self.local_discriminator(local_input, reuse=reuse)

            dout_global = tf.layers.dense(dglobal, 1, kernel_initializer=self.fc_init,
                                          name='dout_global_fc')
            dout_local = dlocal
            # dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            # dout_local = tf.layers.dense(dlocal, 256, name='dout_local_fc')
            # dout_local = tf.reduce_mean(dout_local, axis=1)

            return dout_global, dout_local

    def build_graph_with_losses(self, batch_data, cfg, summary=True, reuse=None):
        # batch_pos = batch_data / 127.5 - 1
        batch_pos = batch_data
        bbox = random_bbox(cfg)
        mask = bbox2mask(bbox, cfg)

        batch_incomplete = batch_pos * (1. - mask)
        ones_x = tf.ones_like(batch_incomplete)[:, :, :, 0:1]
        coarse_network_input = tf.concat([batch_incomplete, ones_x, mask], axis=3)
        # coarse_network_input = tf.concat([batch_incomplete, ones_x, ones_x * mask], axis=3)

        coarse_output = self.coarse_network(coarse_network_input, reuse)
        batch_complete_coarse = coarse_output * mask + batch_pos * (1. - mask)

        # refine_network_input = tf.concat([batch_complete_coarse, ones_x, ones_x * mask], axis=3)
        refine_network_input = tf.concat([batch_complete_coarse, ones_x, mask], axis=3)
        refine_output = self.refine_network(refine_network_input, reuse)
        batch_complete_refine = refine_output * mask + batch_pos * (1. - mask)

        losses = {}

        # local patches
        local_patch_pos = local_patch(batch_pos, bbox)
        local_patch_coarse = local_patch(coarse_output, bbox)
        local_patch_refine = local_patch(refine_output, bbox)
        local_patch_mask = local_patch(mask, bbox)

        l1_alpha = cfg['coarse_l1_alpha']
        losses['coarse_l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_pos - local_patch_coarse) *
                                                             spatial_discounting_mask(cfg))
        losses['coarse_ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - coarse_output) * (1. - mask))

        losses['refine_l1_loss'] = losses['coarse_l1_loss'] + \
            tf.reduce_mean(tf.abs(local_patch_pos - local_patch_refine) *
                           spatial_discounting_mask(cfg))
        losses['refine_ae_loss'] = losses['coarse_ae_loss'] + \
            tf.reduce_mean(tf.abs(batch_pos - refine_output) * (1. - mask))

        losses['coarse_ae_loss'] /= tf.reduce_mean(1. - mask)
        losses['refine_ae_loss'] /= tf.reduce_mean(1. - mask)

        # wgan
        # global discriminator patch
        batch_pos_neg = tf.concat([batch_pos, batch_complete_refine], axis=0)

        # local discriminator patch
        local_patch_pos_neg = tf.concat([local_patch_pos, local_patch_refine], axis=0)

        # wgan with gradient penalty
        pos_neg_global, pos_neg_local = self.build_wgan_discriminator(batch_pos_neg,
                                                                      local_patch_pos_neg,
                                                                      reuse)

        pos_global, neg_global = tf.split(pos_neg_global, 2)
        pos_local, neg_local = tf.split(pos_neg_local, 2)

        # wgan loss
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global)
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local)

        losses['refine_g_loss'] = cfg['global_wgan_loss_alpha'] * g_loss_global + \
            cfg['local_wgan_loss_alpha'] * g_loss_local

        losses['refine_d_loss_global'] = d_loss_global
        losses['refine_d_loss_local'] = d_loss_local
        losses['refine_d_loss'] = losses['refine_d_loss_global'] * 1.4 + losses['refine_d_loss_local'] * 1.4

        # gradient penalty
        interpolates_global = random_interpolates(batch_pos, batch_complete_refine)
        interpolates_local = random_interpolates(local_patch_pos, local_patch_refine)
        dout_global, dout_local = self.build_wgan_discriminator(interpolates_global,
                                                                interpolates_local,
                                                                reuse=True)

        # apply penalty
        # penalty_global = gradient_penalty(interpolates_global, dout_global, mask=mask, norm=750.)
        # penalty_local = gradient_penalty(interpolates_local, dout_local, mask=local_patch_mask, norm=750.)

        # lipschitz penalty
        penalty_global = lipschitz_penalty(interpolates_global, dout_global)
        penalty_local = lipschitz_penalty(interpolates_local, dout_local)

        losses['gp_loss'] = cfg['wgan_gp_lambda'] * (penalty_global + penalty_local)
        losses['refine_d_loss'] += losses['gp_loss']

        g_vars_coarse = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse')
        g_vars_refine = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine')
        g_vars = g_vars_coarse + g_vars_refine
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'wgan_discriminator')

        if summary:
            # stage1
            tf.summary.scalar('rec_loss/coarse_rec_loss', losses['coarse_l1_loss'] + losses['coarse_ae_loss'])
            # tf.summary.scalar('rec_loss/coarse_l1_loss', losses['coarse_l1_loss'])
            # tf.summary.scalar('rec_loss/coarse_ae_loss', losses['coarse_ae_loss'])
            tf.summary.scalar('rec_loss/refine_rec_loss', losses['refine_l1_loss'] + losses['refine_ae_loss'])
            # tf.summary.scalar('rec_loss/refine_l1_loss', losses['refine_l1_loss'])
            # tf.summary.scalar('rec_loss/refine_ae_loss', losses['refine_ae_loss'])

            visual_img = [batch_pos, batch_incomplete, batch_complete_coarse, batch_complete_refine]
            visual_img = tf.concat(visual_img, axis=2)
            images_summary(visual_img, 'raw_incomplete_coarse_refine', 4)

            # stage2
            gradients_summary(g_loss_global, refine_output, name='g_loss_global')
            gradients_summary(g_loss_local, refine_output, name='g_loss_local')

            tf.summary.scalar('convergence/refine_d_loss', losses['refine_d_loss'])
            # tf.summary.scalar('convergence/refine_g_loss', losses['refine_g_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)

            tf.summary.scalar('gradient_penalty/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gradient_penalty/gp_penalty_local', penalty_local)
            tf.summary.scalar('gradient_penalty/gp_penalty_global', penalty_global)

            # summary the magnitude of gradients from different losses w.r.t. predicted image
            # gradients_summary(losses['g_loss'], refine_output, name='g_loss')
            gradients_summary(losses['coarse_l1_loss'] + losses['coarse_ae_loss'],
                              coarse_output,
                              name='rec_loss_grad_to_coarse')
            gradients_summary(losses['refine_l1_loss'] + losses['refine_ae_loss'] + losses['refine_g_loss'],
                              refine_output,
                              name='rec_loss_grad_to_refine')
            gradients_summary(losses['coarse_l1_loss'], coarse_output, name='l1_loss_grad_to_coarse')
            gradients_summary(losses['refine_l1_loss'], refine_output, name='l1_loss_grad_to_refine')
            gradients_summary(losses['coarse_ae_loss'], coarse_output, name='ae_loss_grad_to_coarse')
            gradients_summary(losses['refine_ae_loss'], refine_output, name='ae_loss_grad_to_refine')

        return g_vars, g_vars_coarse, d_vars, losses

    def build_infer_graph(self, batch_data, cfg, bbox=None, name='val'):
        cfg['max_delta_height'] = 0
        cfg['max_delta_width'] = 0

        if bbox is None:
            bbox = random_bbox(cfg)
        mask = bbox2mask(bbox, cfg)

        batch_pos = batch_data
        batch_incomplete = batch_pos * (1. - mask)
        ones_x = tf.ones_like(batch_incomplete)[:, :, :, 0:1]
        coarse_network_input = tf.concat([batch_incomplete, ones_x, mask], axis=3)
        # coarse_network_input = tf.concat([batch_incomplete, ones_x, ones_x * mask], axis=3)

        # inpaint
        coarse_output = self.coarse_network(coarse_network_input, reuse=True)
        batch_complete_coarse = coarse_output * mask + batch_incomplete * (1. - mask)

        # refine_network_input = tf.concat([batch_complete_coarse, ones_x, ones_x * mask], axis=3)
        refine_network_input = tf.concat([batch_complete_coarse, ones_x, mask], axis=3)
        refine_output = self.refine_network(refine_network_input, reuse=True)

        # apply mask and reconstruct
        # batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        batch_complete_coarse = coarse_output * mask + batch_incomplete * (1. - mask)
        batch_complete_refine = refine_output * mask + batch_incomplete * (1. - mask)

        # global image visualization
        visual_img = [batch_pos, batch_incomplete, batch_complete_coarse, batch_complete_refine]
        images_summary(tf.concat(visual_img, axis=2), name + '_raw_incomplete_coarse_refine', 10)

        return (batch_complete_coarse, batch_complete_refine)

    def build_static_infer_graph(self, batch_data, cfg, name):
        bbox = [(tf.constant(cfg['hole_height'] // 2), tf.constant(cfg['hole_width'] // 2),
                 tf.constant(cfg['hole_height']), tf.constant(cfg['hole_width']))]
        bbox = bbox * cfg['batch_size']

        return self.build_infer_graph(batch_data, cfg, bbox, name)

    def build_test_graph(self, batch_data, reuse=None):
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)
        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)

        ones_x = tf.ones_like(batch_incomplete)[:, :, :, 0:1]
        coarse_network_input = tf.concat([batch_incomplete, ones_x, ones_x * masks], axis=3)
        coarse_output = self.coarse_network(coarse_network_input, reuse=reuse)
        batch_complete_coarse = coarse_output * masks + batch_incomplete * (1. - masks)

        refine_network_input = tf.concat([batch_complete_coarse, ones_x, ones_x * masks], axis=3)
        refine_output = self.refine_network(refine_network_input, reuse=reuse)
        batch_complete_refine = refine_output * masks + batch_incomplete * (1. - masks)

        return (batch_incomplete, batch_complete_coarse, batch_complete_refine)


if __name__ == '__main__':
    import yaml
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    model = CompletionModel()
    x = tf.random_uniform([10, 256, 256, 3])

    # coarse = model.coarse_network(x)
    # refine = model.refine_network(coarse)
    # print(coarse.get_shape())
    # print(refine.get_shape())

    g_vars, g_vars_coarse, d_vars, losses = model.build_graph_with_losses(x, cfg)
    print(len(g_vars), len(d_vars))
