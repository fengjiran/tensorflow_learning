import os

import tensorflow as tf


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_mask(height, width, mask_height, mask_width, x=None, y=None):
    top = y if y is not None else tf.random_uniform([], minval=0, maxval=height - mask_height, dtype=tf.int32)
    left = x if x is not None else tf.random_uniform([], minval=0, maxval=width - mask_width, dtype=tf.int32)

    mask = tf.pad(tensor=tf.ones((mask_height, mask_width), dtype=tf.float32),
                  paddings=[[top, height - mask_height - top],
                            [left, width - mask_width - left]])

    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    return mask  # [1, height, width, 1]
