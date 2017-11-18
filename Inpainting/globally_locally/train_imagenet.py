from __future__ import division
from __future__ import print_function

import os
import pickle
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
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models'

# input image size for comletion network and global discrimintor
input_height = 256
input_width = 256

# input image size for local discriminator
gt_height = 128
gt_width = 128

isFirstTimeTrain = False
batch_size = 16

iters_c = 120000  # iters for completion network
iters_d = 2300 * 6  # iters for discriminator
iters_total = 120000 * 6  # total iters

lambda_adv = 0.0004
weight_decay_rate = 0.0001
init_lr = 0.002

# placeholder
is_training = tf.placeholder(tf.bool)

images = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images_with_holes')

masks_c = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='masks_c')

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
adv_all = tf.concat([adv_pos, adv_neg], axis=0)

# print(images.get_shape().as_list())
# print(images_with_hole.get_shape().as_list())
# print(completed_images.get_shape().as_list())
# print(global_dis_inputs_fake.get_shape().as_list())
# print(local_dis_inputs_fake.get_shape().as_list())
# print(local_dis_inputs_real.get_shape().as_list())
# print(adv_pos.get_shape().as_list())
# print(adv_neg.get_shape().as_list())

var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
var_D = tf.get_collection('global_dis_params_conv') +\
    tf.get_collection('global_dis_params_bn') +\
    tf.get_collection('local_dis_params_conv') +\
    tf.get_collection('local_dis_params_bn') +\
    tf.get_collection('combine_dis_params')

loss_recon = tf.reduce_mean(tf.square(masks_c * (images - completed_images)))

loss_adv_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_all,
                                                                    labels=labels_D))
loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_neg,
                                                                    labels=labels_G))

loss_G1 = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))
loss_G2 = loss_recon + lambda_adv * loss_adv_G + weight_decay_rate * \
    tf.reduce_mean(tf.get_collection('weight_decay_gen'))


loss_D = loss_adv_D + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_global_dis') +
                                                         tf.get_collection('weight_decay_local_dis') +
                                                         tf.get_collection('weight_decay_combine_dis'))

opt_g1 = tf.train.AdadeltaOptimizer(learning_rate=init_lr)  # for completion network pre training
opt_g2 = tf.train.AdadeltaOptimizer(learning_rate=init_lr)  # for fine tuning
opt_d = tf.train.AdadeltaOptimizer(learning_rate=init_lr / 10)
# opt_d2 = tf.train.AdadeltaOptimizer()

grads_vars_g1 = opt_g1.compute_gradients(loss=loss_G1,
                                         var_list=var_G)
train_op_g1 = opt_g1.apply_gradients(grads_vars_g1)

grads_vars_g2 = opt_g2.compute_gradients(loss=loss_G2, var_list=var_G)
train_op_g2 = opt_g2.apply_gradients(grads_vars_g2)

grads_vars_d = opt_d.compute_gradients(loss=loss_D, var_list=var_D)
train_op_d = opt_d.apply_gradients(grads_vars_d)

# load the train sample paths
train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]

num_batch = int(len(train_path) / batch_size)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    if not isFirstTimeTrain:
        saver.restore(sess, os.path.join(model_path, 'global_local_consistent'))

        with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
            iters = pickle.load(f)
    else:
        iters = 0
        with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
            pickle.dump(iters, f, protocol=2)
        saver.save(sess, os.path.join(model_path, 'global_local_consistent'))

    while iters < iters_total:
        indx = iters % num_batch
        image_paths = train_path[indx * batch_size:(indx + 1) * batch_size]['image_path'].values
        images_, images_with_hole_, masks_c_, x_locs_, y_locs_ = read_batch(image_paths)

        # a = sess.run(global_dis_inputs_fake, feed_dict={images: images_,
        #                                                 images_with_hole: images_with_hole_,
        #                                                 masks_c: masks_c_,
        #                                                 is_training: False})

        if iters < iters_c:
            _, loss_g1 = sess.run([train_op_g1, loss_G1],
                                  feed_dict={images: images_,
                                             images_with_hole: images_with_hole_,
                                             masks_c: masks_c_,
                                             #  x_locs: x_locs_,
                                             #  y_locs: y_locs_,
                                             is_training: True})
            print('Iter: {0}, loss_g1: {1}'.format(iters, loss_g1))
            iters += 1
        else:
            _, loss_d = sess.run([train_op_d, loss_D],
                                 feed_dict={images: images_,
                                            images_with_hole: images_with_hole_,
                                            masks_c: masks_c_,
                                            x_locs: x_locs_,
                                            y_locs: y_locs_,
                                            is_training: True})
            print('Iter: {0}, loss_d: {1}'.format(iters, loss_d))
            iters += 1

            if iters > iters_c + iters_d:
                _, loss_g2 = sess.run([train_op_g2, loss_G2],
                                      feed_dict={images: images_,
                                                 images_with_hole: images_with_hole_,
                                                 masks_c: masks_c_,
                                                 x_locs: x_locs_,
                                                 y_locs: y_locs_,
                                                 is_training: True})
                print('Iter: {0}, loss_g2: {1}'.format(iters, loss_g2))
                iters += 1

        # with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
        #     pickle.dump(iters, f, protocol=2)

        if iters % 100 == 0:
            with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(model_path, 'global_local_consistent'))


# with tf.Session() as sess:
#     sess.run(init)
#     a, b, c, d = sess.run([completed_images, global_dis_inputs_fake, local_dis_inputs_fake, local_dis_inputs_real],
#                           feed_dict={images: images_,
#                                      images_with_hole: images_with_hole_,
#                                      masks_c: masks_c_,
#                                      x_locs: x_locs_,
#                                      y_locs: y_locs_,
#                                      is_training: True})

#     plt.subplot(241)
#     plt.imshow((255. * (images_[0] + 1) / 2.).astype('uint8'))
#     plt.subplot(242)
#     plt.imshow((255. * (images_with_hole_[0] + 1) / 2.).astype('uint8'))
#     plt.subplot(243)
#     plt.imshow((255. * masks_c_[0]).astype('uint8'))
#     plt.subplot(244)
#     plt.imshow((255. * (a[0] + 1) / 2.).astype('uint8'))
#     plt.subplot(245)
#     plt.imshow((255. * (b[0] + 1) / 2.).astype('uint8'))
#     plt.subplot(246)
#     plt.imshow((255. * (c[0] + 1) / 2.).astype('uint8'))
#     plt.subplot(247)
#     plt.imshow((255. * (d[0] + 1) / 2.).astype('uint8'))

#     plt.show()
