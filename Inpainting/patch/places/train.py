from __future__ import print_function

import os
import platform
import yaml
# import numpy as np
import pandas as pd
import tensorflow as tf
from model import CompletionModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if platform.system() == 'Windows':
    compress_path = cfg['compress_path_win']
    val_path = cfg['val_path_win']
    log_dir = cfg['log_dir_win']
    coarse_model_path = cfg['coarse_model_path_win']
    refine_model_path = cfg['refine_model_path_win']
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = cfg['compress_path_linux_7810']
        val_path = cfg['val_path_linux_7810']
        log_dir = cfg['log_dir_linux_7810']
        coarse_model_path = cfg['coarse_model_path_linux_7810']
        refine_model_path = cfg['refine_model_path_linux_7810']
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = cfg['compress_path_linux_7610']
        val_path = cfg['val_path_linux_7610']
        log_dir = cfg['log_dir_linux_7610']
        coarse_model_path = cfg['coarse_model_path_linux_7610']
        refine_model_path = cfg['refine_model_path_linux_7610']


def input_parse(img_path):
    with tf.device('/cpu:0'):
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        img = tf.cast(img_decoded, tf.float32)
        img = tf.image.resize_image_with_crop_or_pad(img, cfg['img_height'], cfg['img_width'])
        img = img / 127.5 - 1
        return img


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.shuffle(buffer_size=2000)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(cfg['batch_size']))
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
batch_data = iterator.get_next()
# batch_data = tf.image.resize_area(batch_data, [256, 256])
# batch_data = tf.clip_by_value(batch_data, 0., 255.)
# batch_data = batch_data / 127.5 - 1
# print(batch_data.get_shape())

val_filenames = tf.placeholder(tf.string, shape=[None])
val_data = tf.data.Dataset.from_tensor_slices(val_filenames)
val_data = val_data.map(input_parse)
val_data = val_data.batch(cfg['batch_size'])
val_data = val_data.repeat()
val_iterator = val_data.make_initializable_iterator()
val_batch_data = val_iterator.get_next()
# val_batch_data = tf.image.resize_area(val_batch_data, [256, 256])
# val_batch_data = tf.clip_by_value(val_batch_data, 0., 255.)
# val_batch_data = val_batch_data / 127.5 - 1


model = CompletionModel()
# print(batch_data.get_shape())
g_vars, g_vars_coarse, d_vars, losses = model.build_graph_with_losses(batch_data, cfg)


# training settings
# initialize primary trainer
global_step_g = tf.get_variable('global_step_g',
                                [],
                                tf.int32,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
global_step_d = tf.get_variable('global_step_d',
                                [],
                                tf.int32,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
lr_g = cfg['init_lr_g']
lr_d = cfg['init_lr_d']
# lr_g = tf.train.exponential_decay(learning_rate=cfg['init_lr_g'],
#                                   global_step=global_step_g,
#                                   decay_steps=1000,
#                                   decay_rate=0.99)

# lr_d = tf.train.exponential_decay(learning_rate=cfg['init_lr_d'],
#                                   global_step=global_step_d,
#                                   decay_steps=2000,
#                                   decay_rate=0.98)

g_opt = tf.train.AdamOptimizer(lr_g, beta1=0.5, beta2=0.9)
d_opt = tf.train.AdamOptimizer(lr_d, beta1=0.5, beta2=0.9)

coarse_rec_loss = cfg['l1_loss_alpha'] * losses['coarse_l1_loss'] +\
    cfg['ae_loss_alpha'] * losses['coarse_ae_loss']

refine_g_loss = cfg['l1_loss_alpha'] * losses['refine_l1_loss'] +\
    cfg['ae_loss_alpha'] * losses['refine_ae_loss'] +\
    cfg['gan_loss_alpha'] * losses['refine_g_loss']

refine_d_loss = losses['refine_d_loss']

# stage 1
coarse_train = g_opt.minimize(coarse_rec_loss, global_step=global_step_g, var_list=g_vars_coarse)

# stage 2 generator
refine_g_train = g_opt.minimize(refine_g_loss, global_step=global_step_g, var_list=g_vars)

# stage 2 discriminator
refine_d_train = d_opt.minimize(refine_d_loss, global_step=global_step_d, var_list=d_vars)

refine_d_train_ops = []
for i in range(cfg['iteration_d']):
    refine_d_train_ops.append(refine_d_train)
refine_d_train = tf.group(*refine_d_train_ops)

train_path = pd.read_pickle(compress_path)
train_path.index = range(len(train_path))
train_path = train_path[:]['image_path'].values.tolist()
num_batch = len(train_path) // cfg['batch_size']

val_path = pd.read_pickle(val_path)
val_path.index = range(len(val_path))
val_path = val_path[:]['image_path'].values.tolist()

# for _, _, files in os.walk(compress_path):
#     tfrecord_filenames = files
# tfrecord_filenames = [os.path.join(compress_path, file) for file in tfrecord_filenames]

# for _, _, files in os.walk(val_path):
#     val_tfrecord_filenames = files
# val_tfrecord_filenames = [os.path.join(val_path, file) for file in val_tfrecord_filenames]

# num_batch = 29000 // cfg['batch_size']

# print(val_batch_data.get_shape())
if cfg['val']:
    # progress monitor by visualizing static images
    static_inpainted_images = model.build_static_infer_graph(
        val_batch_data,
        cfg,
        'static_images')
    # for i in range(cfg['static_view_num']):
    #     static_fname = val_path[i]
    #     static_image = input_parse(static_fname)
    #     static_image = tf.expand_dims(static_image, 0)
    #     static_inpainted_image = model.build_static_infer_graph(
    #         static_image,
    #         cfg,
    #         'static_view/%d' % i)

# summary
tf.summary.scalar('learning_rate/lr_g', lr_g)
tf.summary.scalar('learning_rate/lr_d', lr_d)
tf.summary.scalar('convergence/refine_g_loss', refine_g_loss)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
all_summary = tf.summary.merge_all()


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(iterator.initializer, feed_dict={filenames: train_path})
    sess.run(val_iterator.initializer, feed_dict={val_filenames: val_path})

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    if cfg['firstTimeTrain']:
        step = 0
        sess.run(tf.global_variables_initializer())
    else:
        # saver.restore(sess, os.path.join(refine_model_path, 'refine_model'))
        saver.restore(sess, os.path.join(coarse_model_path, 'coarse_model'))
        step = global_step_g.eval()

    total_iters = cfg['total_iters']
    while step < total_iters:
        if step < cfg['coarse_iters']:
            # stage 1
            _, loss_value = sess.run([coarse_train, coarse_rec_loss])
            print('Epoch: {}, Iter: {}, coarse_rec_loss: {}'.format(
                int(step / num_batch) + 1,
                step,
                loss_value))
            if (step % 500 == 0) or (step == cfg['coarse_iters'] - 1):
                saver.save(sess, os.path.join(coarse_model_path, 'coarse_model'))
        else:
            # stage 2
            _, _, g_loss, d_loss, d_loss_global, d_loss_local, gp_loss =\
                sess.run([refine_d_train,
                          refine_g_train,
                          refine_g_loss,
                          refine_d_loss,
                          losses['refine_d_loss_global'],
                          losses['refine_d_loss_local'],
                          losses['gp_loss']])
            print('Epoch: {}, Iter: {}, refine_g_loss: {}, refine_d_loss: {}, gp_loss: {}'.format(
                int(step / num_batch) + 1,
                step,
                g_loss,
                d_loss,
                gp_loss))
            print('----------------------refine_d_loss_global: {}, refine_d_loss_local: {}'.format(
                d_loss_global,
                d_loss_local))

            if (step % 500 == 0) or (step == cfg['total_iters'] - 1):
                saver.save(sess, os.path.join(refine_model_path, 'refine_model'))

        if (step % 1000 == 0) or (step == cfg['total_iters'] - 1):
            summary = sess.run(all_summary)
            summary_writer.add_summary(summary, step)

        step += 1
