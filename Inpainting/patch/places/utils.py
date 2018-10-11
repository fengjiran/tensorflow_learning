from __future__ import print_function

import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import ops


def spatial_discounting_mask(cfg):
    gamma = cfg['spatial_discount_gamma']
    height = cfg['hole_height']
    width = cfg['hole_width']
    shape = [1, height, width, 1]

    if cfg['discount_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = gamma**min(i, j, height - i, width - j)

        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
    else:
        mask_values = np.ones(shape)

    return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def random_bbox(cfg):
    # image_shape:(H,W,C)
    height = cfg['img_height']
    width = cfg['img_width']

    hole_height = cfg['hole_height']
    hole_width = cfg['hole_width']

    top = tf.random_uniform([], minval=0, maxval=height - hole_height, dtype=tf.int32)
    left = tf.random_uniform([], minval=0, maxval=width - hole_width, dtype=tf.int32)
    h = tf.constant(hole_height)
    w = tf.constant(hole_width)

    return (top, left, h, w)


def bbox2mask(bbox, cfg):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns
    -------
        tf.Tensor: output with shape [1, H, W, 1]

    """
    height = cfg['img_height']
    width = cfg['img_width']
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
    return tf.reduce_mean(tf.square(slopes - norm) / (norm**2))


def lipschitz_penalty(x, y, mask=None, norm=1.):
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients) * mask, axis=[1, 2, 3]))


def images_summary(images, name, max_outs):
    """Summary images.

    **Note** that images should be scaled to [-1, 1] for 'RGB' or 'BGR',
    [0, 1] for 'GREY'.
    :param images: images tensor (in NHWC format)
    :param name: name of images summary
    :param max_outs: max_outputs for images summary
    :param color_format: 'BGR', 'RGB' or 'GREY'
    :return: None
    """
    # img = tf.cast((images + 1) * 127.5, tf.int8)
    img = (images + 1) / 2.
    tf.summary.image(name, img, max_outs)
    # with tf.variable_scope(name), tf.device('/cpu:0'):
    #     if color_format == 'BGR':
    #         img = tf.clip_by_value(
    #             (tf.reverse(images, [-1]) + 1.) * 127.5, 0., 255.)
    #     elif color_format == 'RGB':
    #         # img = tf.clip_by_value((images + 1.) * 127.5, 0, 255)
    #         # img = (images + 1) / 2
    #         img = tf.cast((img + 1) * 127.5, tf.int8)
    #     elif color_format == 'GREY':
    #         img = tf.clip_by_value(images * 255., 0, 255)
    #     else:
    #         raise NotImplementedError("color format is not supported.")
    #     tf.summary.image(name, img, max_outputs=max_outs)


def gradients_summary(y, x, norm=tf.abs, name='gradients_y_wrt_x'):
    grad = tf.reduce_mean(norm(tf.gradients(y, x)))
    tf.summary.scalar(name, grad)
