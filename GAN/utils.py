from __future__ import division
import os
import math
import random
import pprint
import gzip
import scipy.misc
import numpy as np
from time import gmtime
from time import strftime
from six.moves import xrange
from matplotlib.pyplot import plt
import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_mnist(dataset_name):
    data_dir = os.path.join('./data', dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data
