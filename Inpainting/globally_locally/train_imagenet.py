from __future__ import division
from __future__ import print_function

import platform
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import read_batch
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
# global_step = tf.placeholder(tf.int64)

images = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images_with_holes')

masks_c = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='masks_c')
# ground_truth = tf.placeholder(tf.float32, [batch_size, gt_height, gt_width, 3], name='ground_truth')

x_locs = tf.placeholder(tf.int32, [batch_size])
y_locs = tf.placeholder(tf.int32, [batch_size])

labels_G = tf.ones([batch_size])
labels_D = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)

completed_images = completion_network(images_with_hole, is_training)

# global discriminator inputs
global_dis_inputs_fake = completed_images * masks_c + images_with_hole * (1 - masks_c)
# the global_dis_inputs_real = images

# local discriminator inputs
local_dis_inputs_fake = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0], args[1], args[2], gt_height, gt_width),
                                  elems=(global_dis_inputs_fake, y_locs, x_locs),
                                  dtype=tf.float32)
# crop sample from real samples of 128*128
offset_x = np.random.randint(0, input_width - gt_width)
offset_y = np.random.randint(0, input_height - gt_height)
local_dis_inputs_real = tf.image.crop_to_bounding_box(images, offset_y, offset_x, gt_height, gt_width)

adv_pos = combine_discriminator(global_inputs=images,
                                local_inputs=local_dis_inputs_real,
                                is_training=is_training)
adv_neg = combine_discriminator(global_inputs=global_dis_inputs_fake,
                                local_inputs=local_dis_inputs_fake,
                                is_training=is_training,
                                reuse=True)

print(images.get_shape().as_list())
print(images_with_hole.get_shape().as_list())
print(completed_images.get_shape().as_list())
print(global_dis_inputs_fake.get_shape().as_list())
print(local_dis_inputs_fake.get_shape().as_list())
print(local_dis_inputs_real.get_shape().as_list())
print(adv_pos.get_shape().as_list())
print(adv_neg.get_shape().as_list())

# loss_mse = tf.reduce_mean(tf.square(masks_c * (images - completed_images)))
train_path = pd.read_pickle(compress_path)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
image_paths = train_path[0:batch_size]['image_path'].values

images_, images_with_hole_, masks_c_, x_locs_, y_locs_ = read_batch(image_paths)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    a, b, c, d = sess.run([completed_images, global_dis_inputs_fake, local_dis_inputs_fake, local_dis_inputs_real],
                          feed_dict={images: images_,
                                     images_with_hole: images_with_hole_,
                                     masks_c: masks_c_,
                                     x_locs: x_locs_,
                                     y_locs: y_locs_,
                                     is_training: True})

    plt.subplot(241)
    plt.imshow((255. * (images_[0] + 1) / 2.).astype('uint8'))
    plt.subplot(242)
    plt.imshow((255. * (images_with_hole_[0] + 1) / 2.).astype('uint8'))
    plt.subplot(243)
    plt.imshow((255. * masks_c_[0]).astype('uint8'))
    plt.subplot(244)
    plt.imshow((255. * (completed_images[0] + 1) / 2.).astype('uint8'))
    plt.subplot(245)
    plt.imshow((255. * (global_dis_inputs_fake[0] + 1) / 2.).astype('uint8'))
    plt.subplot(246)
    plt.imshow((255. * (local_dis_inputs_fake[0] + 1) / 2.).astype('uint8'))
    plt.subplot(247)
    plt.imshow((255. * (local_dis_inputs_real[0] + 1) / 2.).astype('uint8'))

    plt.show()


# for t1 in range(iters_total):
#     # load a minibatch of images x from training data
#     # generate masks Mc with random holes for each image x in the minibatch
#     pass
