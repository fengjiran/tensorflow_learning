from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from mpgan_models import completion_network
from mpgan_models import global_discriminator
from mpgan_models import markovian_discriminator

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\celeba_train_path_win.pickle'
    # events_path = config['events_path_win']
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\pretrain_model_global'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/celeba_train_path_linux.pickle'
        # events_path = config['events_path_linux']
        model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/pretrain_model_global'
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = '/home/icie/richard/MPGAN/CelebA/celeba_train_path_linux.pickle'
        # events_path = '/home/icie/richard/MPGAN/CelebA/models_without_adv_l1/events'
        model_path = '/home/icie/richard/MPGAN/CelebA/pretrain_model_global'

# isFirstTimeTrain = False
isFirstTimeTrain = True
batch_size = 16
weight_decay_rate = 1e-4

lr_decay_steps = config['lr_decay_steps']
iters_c = config['iters_c']
iters_d = 10000
# alpha = 0.8
alpha = config['alpha']

init_lr_g = config['init_lr_g']
init_lr_d = config['init_lr_d']
alpha_rec = 1.0
alpha_global = 0
alpha_local = 0

gt_height = 96
gt_width = 96


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

        hole_height = tf.convert_to_tensor(hole_height, tf.float32)
        hole_width = tf.convert_to_tensor(hole_width, tf.float32)
        return ori_image, image_with_hole, mask, x_loc, y_loc, hole_height, hole_width


is_training = tf.placeholder(tf.bool)
global_step_g = tf.get_variable('global_step_g',
                                [],
                                tf.int32,
                                initializer=tf.constant_initializer(0),
                                trainable=False)

global_step_d = tf.get_variable('global_step_d',
                                [],
                                tf.int32,
                                initializer=tf.constant_initializer(0),
                                trainable=False)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)

# a way to prevent the dataset from producing the final batch which has incomplete batch size
# https://github.com/tensorflow/tensorflow/issues/13161
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
# dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

images, images_with_hole, masks, x_locs, y_locs, hole_heights, hole_widths = iterator.get_next()
image_height, image_width = images.get_shape().as_list()[1], images.get_shape().as_list()[2]

syn_images = completion_network(images_with_hole, is_training, batch_size)
# completed_images = (1 - masks) * images + masks * syn_images
completed_images = tf.multiply(1 - masks, images) + tf.multiply(masks, syn_images)

local_dis_inputs_fake = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0],
                                                                                args[1],
                                                                                args[2],
                                                                                gt_height,
                                                                                gt_width),
                                  elems=(completed_images, y_locs, x_locs),
                                  dtype=tf.float32)
local_dis_inputs_real = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0],
                                                                                args[1],
                                                                                args[2],
                                                                                gt_height,
                                                                                gt_width),
                                  elems=(images, y_locs, x_locs),
                                  dtype=tf.float32)

sizes = 3 * tf.multiply(hole_heights, hole_widths)
temp = tf.abs(completed_images - images)
loss_recon = tf.reduce_mean(tf.div(tf.reduce_sum(temp, axis=[1, 2, 3]), sizes))

sizes2 = 3 * (image_height * image_width - tf.multiply(hole_heights, hole_widths))
temp2 = tf.abs((1 - masks) * (syn_images - images))
loss_recon2 = tf.reduce_mean(tf.div(tf.reduce_sum(temp2, axis=[1, 2, 3]), sizes2))

loss_recon = alpha * loss_recon + (1 - alpha) * loss_recon2
# loss_recon = tf.reduce_mean([tf.div(temp[i], sizes[i]) for i in range(batch_size)])
# loss_recon = tf.reduce_mean(tf.abs(completed_images - images))
# loss_recon = tf.reduce_mean(alpha * tf.abs(completed_images - images) +
#                             (1 - alpha) * tf.abs((1 - masks) * (syn_images - images)))

loss_only_g = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))

global_dis_outputs_real = global_discriminator(images, is_training)
global_dis_outputs_fake = global_discriminator(completed_images, is_training, reuse=True)
global_dis_outputs_all = tf.concat([global_dis_outputs_real, global_dis_outputs_fake], axis=0)

local_dis_outputs_real = markovian_discriminator(local_dis_inputs_real, is_training)
local_dis_outputs_fake = markovian_discriminator(local_dis_inputs_fake, is_training, reuse=True)
local_dis_outputs_all = tf.concat([local_dis_outputs_real, local_dis_outputs_fake], axis=0)

labels_global_dis = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)
labels_local_dis = tf.concat([tf.ones_like(local_dis_outputs_real),
                              tf.zeros_like(local_dis_outputs_fake)], axis=0)

loss_global_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=global_dis_outputs_all,
    labels=labels_global_dis
))

loss_local_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=local_dis_outputs_all,
    labels=labels_local_dis
))

loss_global_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=global_dis_outputs_fake,
    labels=tf.ones([batch_size])
))
loss_local_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=local_dis_outputs_fake,
    labels=tf.ones_like(local_dis_outputs_fake)
))

loss_g = alpha_rec * loss_recon + alpha_global * loss_global_g + alpha_local * loss_local_g
loss_d = loss_global_dis  # + loss_local_dis

var_g = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
var_d = tf.get_collection('global_dis_params_conv') + tf.get_collection('global_dis_params_bn')
# tf.get_collection('local_dis_params_conv') +\
# tf.get_collection('global_dis_params_bn') +\
# tf.get_collection('local_dis_params_bn')

lr_g = tf.train.exponential_decay(learning_rate=init_lr_g,
                                  global_step=global_step_g,
                                  decay_steps=lr_decay_steps,
                                  decay_rate=0.98)

lr_d = tf.train.exponential_decay(learning_rate=init_lr_d,
                                  global_step=global_step_d,
                                  decay_steps=lr_decay_steps,
                                  decay_rate=0.97)

opt_g = tf.train.AdamOptimizer(learning_rate=lr_g, beta1=0.5)
opt_d = tf.train.AdamOptimizer(learning_rate=lr_d, beta1=0.5)

# grads and vars
grads_vars_only_g = opt_g.compute_gradients(loss_only_g, var_g)
train_only_g = opt_g.apply_gradients(grads_vars_only_g, global_step_g)

grads_vars_g = opt_g.compute_gradients(loss_g, var_g)
train_g = opt_g.apply_gradients(grads_vars_g, global_step_g)

grads_vars_d = opt_d.compute_gradients(loss_d, var_d)
train_d = opt_d.apply_gradients(grads_vars_d, global_step_d)

# view grads and vars
view_only_g_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_only_g])
view_only_g_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_only_g])

view_g_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_g])
view_g_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_g])

view_d_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_d])
view_d_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_d])

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(decay=0.999)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

train_op_only_g = tf.group(train_only_g, variable_averages_op)
train_op_g = tf.group(train_g, variable_averages_op)
train_op_d = tf.group(train_d, variable_averages_op)

variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)


with tf.Session() as sess:
    train_path = pd.read_pickle(compress_path)
    train_path.index = range(len(train_path))
    train_path = train_path.ix[np.random.permutation(len(train_path))]
    train_path = train_path[:]['image_path'].values.tolist()
    num_batch = int(len(train_path) / batch_size)

    sess.run(iterator.initializer, feed_dict={filenames: train_path})
    sess.run(tf.global_variables_initializer())

    if isFirstTimeTrain:
        iters = 0
        # with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
        #     pickle.dump(iters, f, protocol=2)
        # saver.save(sess, os.path.join(model_path, 'pretrain_model_global'))
    else:
        saver.restore(sess, os.path.join(model_path, 'pretrain_model_global'))
        with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
            iters = pickle.load(f)

    while iters <= iters_c + iters_d:
        if iters <= iters_c:
            _, loss_view_g, gs, lr_view_g = sess.run([train_op_only_g, loss_only_g, global_step_g, lr_g],
                                                     feed_dict={is_training: True})
            print('Epoch: {}, Iter_g: {}, loss_g: {}, lr_g: {}'.format(
                int(iters / num_batch) + 1,
                gs,  # iters,
                loss_view_g,
                lr_view_g))
        else:
            _, loss_view_d, gs, lr_view_d = sess.run([train_op_d, loss_d, global_step_d, lr_d],
                                                     feed_dict={is_training: True})
            print('Epoch: {}, Iter_d: {}, loss_d: {}, lr_d: {}'.format(
                int(iters / num_batch) + 1,
                gs,  # iters,
                loss_view_d,
                lr_view_d))

        if (iters % 200 == 0)or(iters == iters_c + iters_d):
            with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(model_path, 'pretrain_model_global'))

            g_weights_mean, g_grads_mean, d_weights_mean, d_grads_mean = sess.run([view_only_g_weights,
                                                                                   view_only_g_grads,
                                                                                   view_d_weights,
                                                                                   view_d_grads],
                                                                                  feed_dict={is_training: True})
            # summary_writer.add_summary(summary_str, iters)
            print('Epoch: {}, Iter: {}, loss_g: {}, weights_mean: {}, grads_mean: {}'.format(
                int(iters / num_batch) + 1,
                gs,  # iters,
                loss_view_g,
                g_weights_mean,
                g_grads_mean))
            print('-------------------d_weights_mean: {}, d_grads_mean: {}'.format(d_weights_mean,
                                                                                   d_grads_mean))

        iters += 1

    # while iters < iters_c:
    #     _, loss_g, gs, lr_view = sess.run([train_op_only_g, loss_only_g, global_step_g, lr_g],
    #                                       feed_dict={is_training: True})
    #     print('Epoch: {}, Iter: {}, loss_g: {}, lr: {}'.format(
    #         int(iters / num_batch) + 1,
    #         gs,  # iters,
    #         loss_g,
    #         lr_view))

    #     iters += 1

    #     if iters % 200 == 0:
    #         with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
    #             pickle.dump(iters, f, protocol=2)
    #         saver.save(sess, os.path.join(model_path, 'models_without_adv_l1'))

    #         weights_mean, grads_mean = sess.run([view_only_g_weights, view_only_g_grads],
    #                                             feed_dict={is_training: True})
    #         # summary_writer.add_summary(summary_str, iters)
    #         print('Epoch: {}, Iter: {}, loss_g: {}, weights_mean: {}, grads_mean: {}'.format(
    #             int(iters / num_batch) + 1,
    #             gs,  # iters,
    #             loss_g,
    #             weights_mean,
    #             grads_mean))
