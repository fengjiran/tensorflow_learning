from __future__ import division
from __future__ import print_function

import tensorflow as tf


def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
    """
    Define the Inception V1 base architecture.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to.
         can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
         'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
         'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
         'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']

    """
    end_points = {}
