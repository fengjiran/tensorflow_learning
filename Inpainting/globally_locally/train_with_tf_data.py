from __future__ import division
from __future__ import print_function

import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from models import completion_network

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    # model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    # model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l1'

isFirstTimeTrain = True
batch_size = 32
iters_c = 240000  # iters for completion network
lr_decay_steps = 1000
weight_decay_rate = 0.0001
init_lr = 0.001


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 96
        high = 128
        image_height = 256
        image_width = 256
        gt_height = 128
        gt_width = 128

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img = tf.cast(img_decoded, tf.float32)
        img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
        img = 2 * img - 1

        hole_height, hole_width = np.random.randint(low, high, size=(2))
        y = tf.random_uniform([], 0, image_height - hole_height, tf.int32)
        x = tf.random_uniform([], 0, image_width - hole_width, tf.int32)
        # y = np.random.randint(0, image_height - hole_height)
        # x = np.random.randint(0, image_width - hole_width)

        mask = tf.pad(tensor=tf.ones((hole_height, hole_width)),
                      paddings=[[y, image_height - hole_height - y], [x, image_width - hole_width - x]])
        mask = tf.reshape(mask, [image_height, image_width, 1])
        mask = tf.concat([mask] * 3, 2)

        mask_1 = tf.reshape(tensor=mask[:, :, 0] * (2 * 117. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_2 = tf.reshape(tensor=mask[:, :, 1] * (2 * 104. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_3 = tf.reshape(tensor=mask[:, :, 2] * (2 * 123. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_tmp = tf.concat([mask_1, mask_2, mask_3], 2)

        image_with_hole = tf.identity(img)
        image_with_hole = image_with_hole * (1 - mask) + mask_tmp

        # generate the location of 128*128 patch for local discriminator
        x_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, x + hole_width - gt_width]),
                                  maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
                                  dtype=tf.int32)

        y_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, y + hole_height - gt_height]),
                                  maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
                                  dtype=tf.int32)

        # x_loc = np.random.randint(low=max(0, x + hole_width - gt_width),
        #                           high=min(x, image_width - gt_width) + 1)
        # y_loc = np.random.randint(low=max(0, y + hole_height - gt_height),
        #                           high=min(y, image_height - gt_height) + 1)

    return img, image_with_hole, mask, x_loc, y_loc


train_path = pd.read_pickle(compress_path)
# np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
train_path = train_path[:]['image_path'].values.tolist()


filenames = tf.constant(train_path)
# filenames = tf.constant(['E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\test_images\\1_origin.png'])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()

is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int64)

images, images_with_hole, masks, _, _ = iterator.get_next()
completed_images = completion_network(images_with_hole, is_training)
var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
loss_recon = tf.reduce_mean(tf.square(masks * (images - completed_images)))

with tf.Session() as sess:
    a = sess.run(iterator.get_next())
    print(a[2].shape)
    print(a[3], a[4])

    plt.subplot(131)
    plt.imshow((255. * (a[0][0] + 1) / 2.).astype('uint8'))

    plt.subplot(132)
    plt.imshow((255. * (a[0][1] + 1) / 2.).astype('uint8'))

    plt.subplot(133)
    plt.imshow((255. * (a[0][3] + 1) / 2.).astype('uint8'))

    # plt.subplot(133)
    # plt.imshow((255. * a[2]).astype('uint8'))

    plt.show()
