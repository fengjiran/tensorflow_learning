from __future__ import print_function

import os
import platform
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from model import CompletionModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if platform.system() == 'Windows':
    compress_path = cfg['compress_path_win']
    log_dir = cfg['log_dir_win']
    model_path = cfg['model_path_win']
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = cfg['compress_path_linux']
        log_dir = cfg['log_dir_linux']
        model_path = cfg['model_path_linux']
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = '/home/icie/richard/MPGAN/CelebA/celeba_train_path_linux.pickle'
        # events_path = '/home/icie/richard/MPGAN/CelebA/models_without_adv_l1/events'
        # model_path = '/home/icie/richard/MPGAN/CelebA/pretrain_model_global'


def input_parse(img_path):
    with tf.device('/cpu:0'):
        img_height = 218
        img_width = 178
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img = tf.cast(img_decoded, tf.float32)
        img = tf.image.resize_image_with_crop_or_pad(img, img_height, img_width)
        img = tf.image.resize_images(img, [315, 256])
        img = tf.random_crop(img, [cfg['img_height'], cfg['img_width'], 3])
        img = img / 127.5 - 1

        return img


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(cfg['batch_size']))
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

batch_data = iterator.get_next()
# batch_data = tf.image.resize_images(batch_data, [315, 256])
# batch_data = tf.image.random_crop(batch_data, [cfg['img_height'], cfg['img_width']])
# batch_data = batch_data / 127.5 - 1

model = CompletionModel()
g_vars, d_vars, losses = model.build_graph_with_losses(batch_data, cfg)

if cfg['val']:
    pass

# training settings
init_lr = tf.get_variable('lr', shape=[], trainable=False, initializer=tf.constant_initializer(1e-4))

# initialize primary trainer
global_step = tf.get_variable('global_step',
                              [],
                              tf.int32,
                              initializer=tf.zeros_initializer(),
                              trainable=False)
lr = tf.train.exponential_decay(learning_rate=init_lr,
                                global_step=global_step,
                                decay_steps=1000,
                                decay_rate=0.98)
g_opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
d_opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)

coarse_rec_loss = losses['coarse_l1_loss'] + losses['coarse_ae_loss']
refine_g_loss = losses['refine_l1_loss'] + losses['refine_ae_loss'] + losses['refine_g_loss']
refine_d_loss = losses['refine_d_loss']
# g_loss = losses['g_loss']
# d_loss = losses['d_loss']

# stage 1
coarse_grads_vars = g_opt.compute_gradients(coarse_rec_loss, g_vars)
coarse_train = g_opt.apply_gradients(coarse_grads_vars, global_step)

# stage 2 generator
refine_g_grads_vars = g_opt.compute_gradients(refine_g_loss, g_vars)
refine_g_train = g_opt.apply_gradients(refine_g_grads_vars, global_step)

# stage 2 discriminator
refine_d_grads_vars = d_opt.compute_gradients(refine_d_loss, d_vars)
refine_d_train = d_opt.apply_gradients(refine_d_grads_vars, global_step)
refine_d_train_ops = []
for i in range(5):
    refine_d_train_ops.append(refine_d_train)
refine_d_train = tf.group(*refine_d_train_ops)

# summary
tf.summary.scalar('learning_rate', lr)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
all_summary = tf.summary.merge_all()

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load trainset
    train_path = pd.read_pickle(compress_path)
    train_path.index = range(len(train_path))
    train_path = train_path.ix[np.random.permutation(len(train_path))]
    train_path = train_path[:]['image_path'].values.tolist()
    num_batch = int(len(train_path) / cfg['batch_size'])

    sess.run(iterator.initializer, feed_dict={filenames: train_path})
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    if cfg['firstTimeTrain']:
        step = 0
    else:
        saver.restore(sess, os.path.join(model_path, 'model'))
        step = global_step.eval()

    # print(step)

    total_iters = cfg['total_iters']
    while step < total_iters:
        # stage 1
        if step < cfg['coarse_iters']:
            _, loss_value = sess.run([coarse_train, coarse_rec_loss])
            print('Epoch: {}, Iter: {}, coarse_rec_loss: {}'.format(
                int(global_step / num_batch) + 1,
                global_step,
                loss_value))
        else:
            _, _, g_loss, d_loss = sess.run([refine_g_train, refine_d_train, refine_g_loss, refine_d_loss])
            print('Epoch: {}, Iter: {}, refine_g_loss: {}, refine_d_loss: {}'.format(
                int(global_step / num_batch) + 1,
                global_step,
                g_loss,
                d_loss))

        if (step % 5 == 0) or (step == cfg['total_iters'] - 1):
            summary = sess.run(all_summary)
            summary_writer.add_summary(summary, step)

        if (step % 200 == 0) or (step == cfg['total_iters'] - 1):
            saver.save(sess, os.path.join(model_path, 'model'))

        step += 1
