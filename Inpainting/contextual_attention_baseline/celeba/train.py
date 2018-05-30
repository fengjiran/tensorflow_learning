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
    log_dir = cfg['log_dir_win']
    coarse_model_path = cfg['coarse_model_path_win']
    refine_model_path = cfg['refine_model_path_win']
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = cfg['compress_path_linux']
        log_dir = cfg['log_dir_linux']
        coarse_model_path = cfg['coarse_model_path_linux']
        refine_model_path = cfg['refine_model_path_linux']
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
dataset = dataset.shuffle(buffer_size=5000)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(cfg['batch_size']))
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
batch_data = iterator.get_next()

# val_filenames = tf.placeholder(tf.string, shape=[None])
# val_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)
# val_dataset = val_dataset.map(input_parse)
# val_dataset = val_dataset.apply(tf.contrib.data.batch_and_drop_remainder(cfg['batch_size']))
# val_dataset = val_dataset.repeat()
# val_iterator = val_dataset.make_initializable_iterator()
# val_batch_data = val_iterator.get_next()

model = CompletionModel()
g_vars, d_vars, losses = model.build_graph_with_losses(batch_data, cfg)


# training settings
# init_lr_g = tf.get_variable('lr', shape=[], trainable=False,
#                             initializer=tf.constant_initializer(cfg['init_lr_g']))
# init_lr_d = tf.get_variable('lr', shape=[], trainable=False,
#                             initializer=tf.constant_initializer(cfg['init_lr_d']))

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
coarse_train = g_opt.minimize(coarse_rec_loss, global_step=global_step_g, var_list=g_vars)
# coarse_grads_vars = g_opt.compute_gradients(coarse_rec_loss, g_vars)
# coarse_train = g_opt.apply_gradients(coarse_grads_vars, global_step_g)

# stage 2 generator
refine_g_train = g_opt.minimize(refine_g_loss, global_step=global_step_g, var_list=g_vars)
# refine_g_grads_vars = g_opt.compute_gradients(refine_g_loss, g_vars)
# refine_g_train = g_opt.apply_gradients(refine_g_grads_vars, global_step_g)

# stage 2 discriminator
refine_d_train = d_opt.minimize(refine_d_loss, global_step=global_step_d, var_list=d_vars)
# refine_d_grads_vars = d_opt.compute_gradients(refine_d_loss, d_vars)
# refine_d_train = d_opt.apply_gradients(refine_d_grads_vars, global_step_d)
# refine_d_train = d_opt.apply_gradients(refine_d_grads_vars, global_step)

refine_d_train_ops = []
for i in range(5):
    refine_d_train_ops.append(refine_d_train)
refine_d_train = tf.group(*refine_d_train_ops)

# load trainset and validation set
data_path = pd.read_pickle(compress_path)
data_path.index = range(len(data_path))
# train_path = train_path.ix[np.random.permutation(len(train_path))]
data_path = data_path[:]['image_path'].values.tolist()
train_path = data_path[0:182637]
val_path = data_path[182638:]
num_batch = int(len(train_path) / cfg['batch_size'])

if cfg['val']:
    # progress monitor by visualizing static images
    for i in range(cfg['static_view_num']):
        static_fname = val_path[i]
        static_image = input_parse(static_fname)
        static_image = tf.expand_dims(static_image, 0)
        static_inpainted_image = model.build_static_infer_graph(
            static_image,
            cfg,
            'static_view/%d' % i)

# summary
tf.summary.scalar('learning_rate/lr_g', lr_g)
tf.summary.scalar('learning_rate/lr_d', lr_d)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
all_summary = tf.summary.merge_all()


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(iterator.initializer, feed_dict={filenames: train_path})

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
            if (step % 200 == 0) or (step == cfg['coarse_iters'] - 1):
                saver.save(sess, os.path.join(coarse_model_path, 'coarse_model'))
        else:
            # stage 2
            _, _, g_loss, d_loss = sess.run([refine_g_train, refine_d_train, refine_g_loss, refine_d_loss])
            print('Epoch: {}, Iter: {}, refine_g_loss: {}, refine_d_loss: {}'.format(
                int(step / num_batch) + 1,
                step,
                g_loss,
                d_loss))
            if (step % 200 == 0) or (step == cfg['total_iters'] - 1):
                saver.save(sess, os.path.join(refine_model_path, 'refine_model'))

        if (step % 50 == 0) or (step == cfg['total_iters'] - 1):
            summary = sess.run(all_summary)
            summary_writer.add_summary(summary, step)

        step += 1
