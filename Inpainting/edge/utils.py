import os
import sys
import time
import numpy as np
import tensorflow as tf


def create_mask(height, width, mask_height, mask_width, x=None, y=None):
    top = y if y is not None else tf.random_uniform([], minval=0, maxval=height - mask_height, dtype=tf.int32)
    left = x if x is not None else tf.random_uniform([], minval=0, maxval=width - mask_width, dtype=tf.int32)

    mask = tf.pad(tensor=tf.ones((mask_height, mask_width), dtype=tf.float32),
                  paddings=[[top, height - mask_height - top],
                            [left, width - mask_width - left]])

    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    return mask  # [1, height, width, 1]


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
    img = (images + 1) / 2.
    tf.summary.image(name, img, max_outs)
