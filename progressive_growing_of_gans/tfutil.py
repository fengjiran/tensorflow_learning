# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import inspect
import importlib
import imp
import numpy as np
from collections import OrderedDict
import tensorflow as tf


def run(*args, **kwargs):  # Run the specified ops in the default session.
    return tf.get_default_session().run(*args, **kwargs)


def is_tf_expression(x):
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))
    # return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)


def shape_to_list(shape):
    return [dim.value for dim in shape]


def flatten(x):
    with tf.name_scope('Flatten'):
        return tf.reshape(x, [-1])


def log2(x):
    with tf.name_scope('Log2'):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))


def exp2(x):
    with tf.name_scope('Exp2'):
        return tf.exp(x * np.float32(np.log(2.0)))


def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t


def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def absolute_name_scope(scope):  # Forcefully enter the specified name scope, ignoring any surrounding scopes.
    return tf.name_scope(scope + '/')

# Initialize TensorFlow graph and session using good default settings.


def init_tf(config_dict=dict()):
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)
