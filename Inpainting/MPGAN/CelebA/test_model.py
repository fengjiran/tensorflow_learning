from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import FCLayer


def completion_network(images, is_training, batch_size):
    """Construct completion network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    cnum = 32
    input_channel = images.get_shape().as_list()[3]

    with tf.variable_scope('generator'):
        conv1 = Conv2dLayer(images, [5, 5, input_channel, cnum], stride=1, name='conv1')
        conv2 = Conv2dLayer(conv1.output, [3, 3, cnum, 2 * cnum], stride=2, name='conv2')
