from __future__ import division
from __future__ import print_function

import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_image
from utils import crop_image_with_hole
from utils import array_to_image

from models import completion_network
from models import combine_discriminator

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'

isFirstTimeTrain = True
batch_size = 96

iters_c = 20000  # iters for completion network
iters_d = 2300  # iters for discriminator
iters_total = 120000  # total iters

# placeholder
is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int64)
images = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], name='images')
ground_truth = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='ground_truth')

train_path = pd.read_pickle(compress_path)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
