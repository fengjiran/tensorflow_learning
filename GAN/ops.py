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
