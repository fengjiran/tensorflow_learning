from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from models import completion_network


if platform.system() == 'Windows':
    compress_path = compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'

elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'


isFirstTimeTrain = True
batch_size = 64


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 48
        high = 96
        image_height = 178
        image_width = 178
        gt_height = 96
        gt_width = 96

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        img = tf.cast(img_decoded, tf.float32)
        img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)

        # input image range from -1 to 1
        img = 2 * img - 1

        ori_image = tf.identity(img)

        hole_height, hole_width = np.random.randint(low, high, size=(2))
        y = tf.random_uniform([], 0, image_height - hole_height, tf.int32)
        x = tf.random_uniform([], 0, image_width - hole_width, tf.int32)

        mask = tf.pad(tensor=tf.ones((hole_height, hole_width)),
                      paddings=[[y, image_height - hole_height - y],
                                [x, image_width - hole_width - x]])
        mask = tf.reshape(mask, [image_height, image_width, 1])
        mask = tf.concat([mask] * 3, 2)

        image_with_hole = img * (1 - mask) + mask

        # generate the location of 96*96 patch for local discriminator
        x_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, x + hole_width - gt_width]),
                                  maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
                                  dtype=tf.int32)
        y_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, y + hole_height - gt_height]),
                                  maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
                                  dtype=tf.int32)

        return ori_image, image_with_hole, mask, x_loc, y_loc


is_training = tf.placeholder(tf.bool)
global_step = tf.get_variable('global_step',
                              [],
                              tf.int32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
filenames = tf.placeholder(tf.string, shape=[None])

dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

images, images_with_hole, masks, _, _ = iterator.get_next()
completed_images = completion_network(images_with_hole, is_training, batch_size)
var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')

with tf.Session() as sess:
    train_path = pd.read_pickle(compress_path)
    train_path.index = range(len(train_path))
    train_path = train_path.ix[np.random.permutation(len(train_path))]
    train_path = train_path[:]['image_path'].values.tolist()
    num_batch = int(len(train_path) / batch_size)
    print(num_batch)
