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

# ----------------------------------------------------------------------------
# Create tf.Session based on config dict of the form
# {'gpu_options.allow_growth': True}


def create_session(config_dict=dict(), force_as_default=False):
    config = tf.ConfigProto()
    for key, value in config_dict.items():
        fields = key.split('.')
        obj = config
        for field in fields[:-1]:
            obj = getattr(obj, field)
        setattr(obj, fields[-1], value)
    session = tf.Session(config=config)
    if force_as_default:
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session

# ----------------------------------------------------------------------------
# Initialize all tf.Variables that have not already been initialized.
# Equivalent to the following, but more efficient and does not bloat the tf graph:
# tf.variables_initializer(tf.report_unitialized_variables()).run()


def init_uninited_vars(vars=None):
    if vars is None:
        vars = tf.global_variables()
    test_vars = []
    test_ops = []
    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in vars:
            assert is_tf_expression(var)
            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/IsVariableInitialized:0'))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)
                with absolute_name_scope(var.name.split(':')[0]):
                    test_ops.append(tf.is_variable_initialized(var))
    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])
