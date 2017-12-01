from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt

from utils import read_batch
from models import completion_network

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2'

# input image size for comletion network and global discrimintor
input_height = 256
input_width = 256

isFirstTimeTrain = False
batch_size = 32

iters_c = 120000  # iters for completion network

lr_decay_steps = 1000
weight_decay_rate = 0.0001
init_lr = 0.001

# placeholder
is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int64)
images = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images_with_holes')

masks_c = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='masks_c')

completed_images = completion_network(images_with_hole, is_training)

var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')

loss_recon = tf.reduce_mean(tf.square(masks_c * (images - completed_images)))
loss_G = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))

lr = tf.train.exponential_decay(learning_rate=init_lr,
                                global_step=global_step,
                                decay_steps=lr_decay_steps,
                                decay_rate=0.992)

opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
# opt = tf.train.GradientDescentOptimizer(lr)
grads_vars_g = opt.compute_gradients(loss_G, var_G)
# grads_vars_g = [(tf.clip_by_value(gv[0], -10., 10.), gv[1]) for gv in grads_vars_g]
train_op_g = opt.apply_gradients(grads_vars_g)

grads = tf.gradients(ys=loss_G, xs=var_G)
grads = [tf.clip_by_value(grad, -10., 10.) if grad is not None else tf.zeros_like(var)
         for (grad, var) in zip(grads, var_G)]
view_grads = tf.reduce_mean([tf.reduce_mean(grad) if grad is not None else 0. for grad in grads])
# grads[4] = tf.clip_by_value(grads[4], -1., 1.)
print(len(grads))
print(grads[0].get_shape())
print(grads[0])

# load the train sample paths
train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))  # 1807854
train_path = train_path.ix[np.random.permutation(len(train_path))]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    indx = 2
    image_paths = train_path[indx * batch_size:(indx + 1) * batch_size]['image_path'].values
    images_, images_with_hole_, masks_c_, x_locs_, y_locs_ = read_batch(image_paths)
    a, b = sess.run((grads[0], view_grads),
                    feed_dict={images: images_,
                               images_with_hole: images_with_hole_,
                               masks_c: masks_c_,
                               is_training: True})
    print(a)
    print(b)


# num_batch = int(len(train_path) / batch_size)

# saver = tf.train.Saver()
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)

#     if not isFirstTimeTrain:
#         saver.restore(sess, os.path.join(model_path, 'models_without_adv_l2'))

#         with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
#             iters = pickle.load(f)
#     else:
#         iters = 0
#         with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
#             pickle.dump(iters, f, protocol=2)
#         saver.save(sess, os.path.join(model_path, 'models_without_adv_l2'))

#     while iters < iters_c:
#         indx = iters % num_batch
#         image_paths = train_path[indx * batch_size:(indx + 1) * batch_size]['image_path'].values
#         images_, images_with_hole_, masks_c_, x_locs_, y_locs_ = read_batch(image_paths)
#         _, loss_g = sess.run([train_op_g, loss_G],
#                              feed_dict={images: images_,
#                                         images_with_hole: images_with_hole_,
#                                         masks_c: masks_c_,
#                                         global_step: iters,
#                                         is_training: True})

#         print('Epoch: {}, Iter: {}, loss_g: {}'.format(int(iters / num_batch) + 1, iters, loss_g))
#         iters += 1

#         if iters % 100 == 0:
#             with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
#                 pickle.dump(iters, f, protocol=2)
#             saver.save(sess, os.path.join(model_path, 'models_without_adv_l2'))
