import math
import numpy as np
import tensorflow as tf


def batch_norm(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(x, name='instance_norm'):
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable('scale', [depth],
                                initializer=tf.random_normal_initializer(1.0, 0.02))
