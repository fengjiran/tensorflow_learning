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
    return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)
