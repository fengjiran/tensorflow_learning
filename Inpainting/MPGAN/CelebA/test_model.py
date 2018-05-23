from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
# from utils import FCLayer


def coarse_network(images, batch_size):
    """Construct coarse network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    cnum = 32
    input_channel = images.get_shape().as_list()[3]

    with tf.variable_scope('coarse'):
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
                             output_shape=[batch_size, conv3.output_shape[1],
                                           conv3.output_shape[2], 2 * cnum],
                             stride=1,
                             name='conv13_upsample')
        conv14 = Conv2dLayer(tf.nn.elu(conv13.output), [3, 3, 2 * cnum, 2 * cnum], stride=1, name='conv14')
        conv15 = DeconvLayer(inputs=tf.nn.elu(conv14.output),
                             filter_shape=[3, 3, cnum, 2 * cnum],
                             output_shape=[batch_size, conv1.output_shape[1],
                                           conv1.output_shape[2], cnum],
                             stride=1,
                             name='conv15_upsample')
        conv16 = Conv2dLayer(tf.nn.elu(conv15.output), [3, 3, cnum, int(cnum / 2)], stride=1, name='conv16')
        conv17 = Conv2dLayer(tf.nn.elu(conv16.output), [3, 3, int(cnum / 2), 3], stride=1, name='conv17')
        conv_output = tf.clip_by_value(conv17.output, -1., 1.)

        for i in range(1, 18):
            conv_layers.append(eval('conv{}'.format(i)))

        for conv in conv_layers:
            print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.output_shape))

        return conv_output


def refine_network(images, batch_size):
    """Construct refine network."""
    conv_layers = []
    cnum = 32
    input_channel = images.get_shape().as_list()[3]

    with tf.variable_scope('refine'):
        conv1 = Conv2dLayer(images, [5, 5, input_channel, cnum], stride=1, name='conv1')
        conv2 = Conv2dLayer(tf.nn.elu(conv1.output), [3, 3, cnum, cnum], stride=2, name='conv2_downsample')
        conv3 = Conv2dLayer(tf.nn.elu(conv2.output), [3, 3, cnum, 2 * cnum], stride=1, name='conv3')
        conv4 = Conv2dLayer(tf.nn.elu(conv3.output), [3, 3, 2 * cnum, 2 * cnum], stride=2, name='conv4_downsample')
        conv5 = Conv2dLayer(tf.nn.elu(conv4.output), [3, 3, 2 * cnum, 4 * cnum], stride=1, name='conv5')
        conv6 = Conv2dLayer(tf.nn.elu(conv5.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv6')

        conv7 = DilatedConv2dLayer(tf.nn.elu(conv6.output), [3, 3, 4 * cnum, 4 * cnum], rate=2, name='conv7_atrous')
        conv8 = DilatedConv2dLayer(tf.nn.elu(conv7.output), [3, 3, 4 * cnum, 4 * cnum], rate=4, name='conv8_atrous')
        conv9 = DilatedConv2dLayer(tf.nn.elu(conv8.output), [3, 3, 4 * cnum, 4 * cnum], rate=8, name='conv9_atrous')
        conv10 = DilatedConv2dLayer(tf.nn.elu(conv9.output), [3, 3, 4 * cnum, 4 * cnum], rate=16, name='conv10_atrous')

        conv11 = Conv2dLayer(tf.nn.elu(conv10.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv11')
        conv12 = Conv2dLayer(tf.nn.elu(conv11.output), [3, 3, 4 * cnum, 4 * cnum], stride=1, name='conv12')

        conv13 = DeconvLayer(inputs=tf.nn.elu(conv12.output),
                             filter_shape=[3, 3, 2 * cnum, 4 * cnum],
                             output_shape=[batch_size, conv3.output_shape[1],
                                           conv3.output_shape[2], 2 * cnum],
                             stride=1,
                             name='conv13_upsample')
        conv14 = Conv2dLayer(tf.nn.elu(conv13.output), [3, 3, 2 * cnum, 2 * cnum], stride=1, name='conv14')

        conv15 = DeconvLayer(inputs=tf.nn.elu(conv14.output),
                             filter_shape=[3, 3, cnum, 2 * cnum],
                             output_shape=[batch_size, conv1.output_shape[1],
                                           conv1.output_shape[2], cnum],
                             stride=1,
                             name='conv15_upsample')

        conv16 = Conv2dLayer(tf.nn.elu(conv15.output), [3, 3, cnum, int(cnum / 2)], stride=1, name='conv16')
        conv17 = Conv2dLayer(tf.nn.elu(conv16.output), [3, 3, int(cnum / 2), 3], stride=1, name='conv17')
        conv_output = tf.clip_by_value(conv17.output, -1., 1.)

        for i in range(1, 18):
            conv_layers.append(eval('conv{}'.format(i)))

        for conv in conv_layers:
            print('conv:{}, output_shape:{}'.format(conv_layers.index(conv) + 1, conv.output_shape))

        return conv_output


def global_discriminator(x, reuse=None):
    cnum = 64
    input_channel = x.get_shape().as_list()[3]
    with tf.variable_scope('global_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(x, [5, 5, input_channel, cnum], stride=2, name='conv1')
        conv2 = Conv2dLayer(tf.nn.leaky_relu(conv1.output), [5, 5, cnum, 2 * cnum], stride=2, name='conv2')
        conv3 = Conv2dLayer(tf.nn.leaky_relu(conv2.output), [5, 5, 2 * cnum, 4 * cnum], stride=2, name='conv3')
        conv4 = Conv2dLayer(tf.nn.leaky_relu(conv3.output), [5, 5, 4 * cnum, 4 * cnum], stride=2, name='conv4')

        return tf.contrib.layers.flatten(tf.nn.leaky_relu(conv4.output))


def local_discriminator(x, reuse=None):
    cnum = 64
    input_channel = x.get_shape().as_list()[3]
    with tf.variable_scope('local_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(x, [5, 5, input_channel, cnum], stride=2, name='conv1')
        conv2 = Conv2dLayer(tf.nn.leaky_relu(conv1.output), [5, 5, cnum, 2 * cnum], stride=2, name='conv2')
        conv3 = Conv2dLayer(tf.nn.leaky_relu(conv2.output), [5, 5, 2 * cnum, 4 * cnum], stride=2, name='conv3')
        conv4 = Conv2dLayer(tf.nn.leaky_relu(conv3.output), [5, 5, 4 * cnum, 8 * cnum], stride=2, name='conv4')

        return tf.contrib.layers.flatten(tf.nn.leaky_relu(conv4.output))


def build_wgan_discriminator(global_input, local_input, reuse=None):
    with tf.variable_scope('wgan_discriminator', reuse=reuse):
        dglobal = global_discriminator(global_input, reuse=reuse)
        dlocal = local_discriminator(local_input, reuse=reuse)

        dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
        dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')

        return dout_global, dout_local


def build_graph_with_losses(batch_data,
                            image_shape,
                            hole_height,
                            hole_width,
                            pretrain_coarse=True,
                            reuse=None,
                            summary=False):
    l1_alpha = 1.2
    global_wgan_loss_alpha = 1.
    wgan_gp_lambda = 10
    gan_loss_alpha = 0.001
    l1_loss_alpha = 1.2
    ae_loss_alpha = 1.2
    ae_loss = True

    batch_size = batch_data.get_shape().as_list()[0]
    batch_pos = batch_data / 127.5 - 1
    bbox = random_bbox(image_shape, hole_height, hole_width)
    mask = bbox2mask(image_shape, bbox)  # (1,height,width,1)
    batch_incomplete = batch_pos * (1. - mask)
    ones_x = tf.ones_like(batch_incomplete)[:, :, :, 0:1]
    x = tf.concat([batch_incomplete, ones_x, ones_x * mask], axis=3)

    coarse_output = coarse_network(x, batch_size)

    # apply mask and complete image
    batch_complete_coarse = coarse_output * mask + batch_incomplete * (1. - mask)
    refine_network_input = tf.concat([batch_complete_coarse, ones_x, ones_x * mask], axis=3)
    refine_output = refine_network(refine_network_input, batch_size)

    if pretrain_coarse:
        batch_predicted = coarse_output
    else:
        batch_predicted = refine_output

    batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)

    # local patches
    local_patch_batch_pos = local_patch(batch_pos, bbox)
    local_patch_batch_predicted = local_patch(batch_predicted, bbox)
    local_patch_coarse = local_patch(coarse_output, bbox)
    local_patch_refine = local_patch(refine_output, bbox)
    local_patch_batch_complete = local_patch(batch_complete, bbox)
    local_patch_mask = local_patch(mask, bbox)

    losses = {}
    losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_coarse) *
                                                  spatial_discounting_mask(0.9, hole_height, hole_width))

    if not pretrain_coarse:
        losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_refine) *
                                            spatial_discounting_mask(0.9, hole_height, hole_width))

    losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - coarse_output) * (1. - mask))
    if not pretrain_coarse:
        losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - refine_output) * (1. - mask))
    losses['ae_loss'] /= tf.reduce_mean(1. - mask)  # good idea

    # if summary:
    #     scalar_summary('losses/l1_loss', losses['l1_loss'])
    #     scalar_summary('losses/ae_loss', losses['ae_loss'])
    #     viz_img = [batch_pos, batch_incomplete, batch_complete]

    #     images_summary(tf.concat(viz_img, axis=2), 'raw_incomplete_predicted_complete', 10)

    # gan
    batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

    # local deteminator patch
    local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], axis=0)

    # wgan with gradient penalty
    pos_neg_global, pos_neg_local = build_wgan_discriminator(batch_pos_neg, local_patch_batch_pos_neg, reuse)
    pos_global, neg_global = tf.split(pos_neg_global, 2)
    pos_local, neg_local = tf.split(pos_neg_local, 2)

    # wgan loss
    g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global)
    g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local)

    losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
    losses['d_loss'] = d_loss_global + d_loss_local

    # gradient penalty
    interpolates_global = random_interpolates(batch_pos, batch_complete)
    interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
    dout_global, dout_local = build_wgan_discriminator(interpolates_global, interpolates_local, reuse=True)

    # apply penalty
    penalty_global = gradient_penalty(interpolates_global, dout_global, mask=mask)
    penalty_local = gradient_penalty(interpolates_local, dout_local, mask=local_patch_mask)

    losses['gp_loss'] = wgan_gp_lambda * (penalty_global + penalty_local)
    losses['d_loss'] = losses['d_loss'] + losses['gp_loss']

    if pretrain_coarse:
        losses['g_loss'] = 0
    else:
        losses['g_loss'] = gan_loss_alpha * losses['g_loss']

    losses['g_loss'] += l1_loss_alpha * losses['l1_loss']

    if ae_loss:
        losses['g_loss'] += ae_loss_alpha * losses['ae_loss']

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse') +\
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'wgan_discriminator')

    return g_vars, d_vars, losses


def spatial_discounting_mask(gamma, height, width):
    shape = [1, height, width, 1]
    mask_values = np.ones((height, width))

    for i in range(height):
        for j in range(width):
            mask_values[i, j] = gamma**min(i, j, height - i, width - j)

    mask_values = np.expand_dims(mask_values, 0)
    mask_values = np.expand_dims(mask_values, 3)

    return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def random_bbox(image_shape, hole_height, hole_width):
    height = image_shape[0]
    width = image_shape[1]

    top = tf.random_uniform([], minval=0, maxval=height - hole_height, dtype=tf.int32)
    left = tf.random_uniform([], minval=0, maxval=width - hole_width, dtype=tf.int32)
    h = tf.constant(hole_height)
    w = tf.constant(hole_width)

    return (top, left, h, w)


def bbox2mask(image_shape, bbox):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns
    -------
        tf.Tensor: output with shape [1, H, W, 1]

    """
    height = image_shape[0]
    width = image_shape[1]
    top, left, h, w = bbox

    mask = tf.pad(tensor=tf.ones((h, w), dtype=tf.float32),
                  paddings=[[top, height - h - top],
                            [left, width - w - left]])

    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns
    -------
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def gan_wgan_loss(pos, neg):
    d_loss = tf.reduce_mean(neg - pos)
    g_loss = -tf.reduce_mean(neg)

    return g_loss, d_loss


def random_interpolates(x, y, alpha=None):
    """Generate.

    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])

    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha * (y - x)
    return tf.reshape(interpolates, shape)


def gradient_penalty(x, y, mask=None, norm=1.):
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


if __name__ == '__main__':
    x = tf.random_uniform([10, 256, 256, 3])
    # y = refine_network(x, 10)
    image_shape = (256, 256, 3)
    bbox = (5, 5, 128, 128)
    # mask = bbox2mask(image_shape, bbox)
    y = local_patch(x, bbox)
    print(y.get_shape())
