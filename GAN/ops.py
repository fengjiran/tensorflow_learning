import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def bn(x, is_training):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-5,
                                         training=is_training)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
