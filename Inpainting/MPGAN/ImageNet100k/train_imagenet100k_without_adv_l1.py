from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from mpgan_models import completion_network


if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ImageNet100k\\imagenet100k_train_path_win.pickle'
    events_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ImageNet100k\\models_without_adv_l1\\events'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ImageNet100k\\models_without_adv_l1'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ImageNet100k/imagenet100k_train_path_linux_7810.pickle'
        events_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ImageNet100k/models_without_adv_l1/events'
        model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ImageNet100k/models_without_adv_l1'
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = '/home/icie/richard/MPGAN/ImageNet100k/imagenet100k_train_path_linux_7610.pickle'
        events_path = '/home/icie/richard/MPGAN/ImageNet100k/models_without_adv_l1/events'
        model_path = '/home/icie/richard/MPGAN/ImageNet100k/models_without_adv_l1'

# isFirstTimeTrain = False
isFirstTimeTrain = True
batch_size = 16
weight_decay_rate = 1e-4
init_lr = 3e-4
lr_decay_steps = 1000
iters_c = 10 * int(100000 / batch_size)  # 10 epochs
alpha = 0.7


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 96
        high = 128
        image_height = 256
        image_width = 256
        # gt_height = 150
        # gt_width = 150

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        img = tf.cast(img_decoded, tf.float32)
        # img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)

        # input image range from -1 to 1
        img = img / 127.5 - 1

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

        # generate the location of 300*300 patch for local discriminator
        # x_loc = tf.random_uniform(shape=[],
        #                           minval=tf.reduce_max([0, x + hole_width - gt_width]),
        #                           maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
        #                           dtype=tf.int32)
        # y_loc = tf.random_uniform(shape=[],
        #                           minval=tf.reduce_max([0, y + hole_height - gt_height]),
        #                           maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
        #                           dtype=tf.int32)

        return ori_image, image_with_hole, mask  # , x_loc, y_loc


is_training = tf.placeholder(tf.bool)
global_step = tf.get_variable('global_step',
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

images, images_with_hole, masks = iterator.get_next()
syn_images = completion_network(images_with_hole, is_training, batch_size)
completed_images = (1 - masks) * images + masks * syn_images
# loss_recon = tf.reduce_mean(tf.nn.l2_loss(completed_images - images))
# loss_recon = tf.reduce_mean(tf.square(completed_images - images))
loss_recon = tf.reduce_mean(alpha * tf.abs(completed_images - images) +
                            (1 - alpha) * tf.abs((1 - masks) * (syn_images - images)))
loss_G = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))
var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')

summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

# Add a summary to track the loss.
summaries.append(tf.summary.scalar('generator_loss', loss_G))

lr = tf.train.exponential_decay(learning_rate=init_lr,
                                global_step=global_step,
                                decay_steps=lr_decay_steps,
                                decay_rate=0.99)

# Add a summary to track the learning rate.
summaries.append(tf.summary.scalar('learning_rate', lr))

opt = tf.train.AdamOptimizer(lr, beta1=0.5)
grads_vars_g = opt.compute_gradients(loss_G, var_G)

# Add histograms for gradients.
for grad, var in grads_vars_g:
    if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

train_op_g = opt.apply_gradients(grads_vars_g, global_step)

# Add histograms for trainable variables.
for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

train_op = tf.group(train_op_g, variable_averages_op)

view_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0.
                             for gv in grads_vars_g])
view_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_g])

variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
summary_op = tf.summary.merge(summaries)
summary_writer = tf.summary.FileWriter(events_path)

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
        with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
            pickle.dump(iters, f, protocol=2)
        saver.save(sess, os.path.join(model_path, 'models_without_adv_l1'))
    else:
        saver.restore(sess, os.path.join(model_path, 'models_without_adv_l1'))
        with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
            iters = pickle.load(f)

    while iters < iters_c:
        _, loss_g, gs, lr_view = sess.run([train_op, loss_G, global_step, lr],
                                          feed_dict={is_training: True})
        print('Epoch: {}, Iter: {}, loss_g: {}, lr: {}'.format(
            int(iters / num_batch) + 1,
            gs,  # iters,
            loss_g,
            lr_view))

        iters += 1

        if iters % 100 == 0:
            with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(model_path, 'models_without_adv_l1'))

            summary_str, weights_mean, grads_mean = sess.run([summary_op, view_weights, view_grads],
                                                             feed_dict={is_training: True})
            summary_writer.add_summary(summary_str, iters)
            print('Epoch: {}, Iter: {}, loss_g: {}, weights_mean: {}, grads_mean: {}'.format(
                int(iters / num_batch) + 1,
                gs,  # iters,
                loss_g,
                weights_mean,
                grads_mean))
