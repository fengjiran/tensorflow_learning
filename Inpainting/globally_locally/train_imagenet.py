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

# input image size for comletion network and global discrimintor
input_height = 256
input_width = 256

# input image size for local discriminator
gt_height = 128
gt_width = 128

isFirstTimeTrain = True
batch_size = 96

iters_c = 20000  # iters for completion network
iters_d = 2300  # iters for discriminator
iters_total = 120000  # total iters

# placeholder
is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int64)

images = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images_with_holes')

masks = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='masks')
ground_truth = tf.placeholder(tf.float32, [batch_size, gt_height, gt_width, 3], name='ground_truth')

# hole_height = tf.placeholder(tf.int32)
# hole_width = tf.placeholder(tf.int32)
# y_init = tf.placeholder(tf.int32)
# x_init = tf.placeholder(tf.int32)

# # mask for every image
# mask = tf.pad(tensor=tf.ones([hole_height, hole_width]),
#               paddings=[[input_height - hole_height - y_init, y_init], [x_init, input_width - hole_width - x_init]])
# mask = tf.reshape(mask, [input_height, input_width, 1])
# mask = tf.concat([mask] * 3, 2)

completed_images = completion_network(images_with_hole, is_training)

loss_mse = tf.reduce_mean(tf.square(masks * (images - completed_images)))

train_path = pd.read_pickle(compress_path)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]

for t1 in range(iters_total):
    # load a minibatch of images x from training data
    # generate masks Mc with random holes for each image x in the minibatch
    x = 0
