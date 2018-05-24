from __future__ import print_function

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
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img = tf.cast(img_decoded, tf.float32)
        img = tf.image.resize_images(img, [315, 256])
        img = tf.image.random_crop(img, [cfg['img_height'], cfg['img_width']])
        img = img / 127.5 - 1

        return img


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(cfg['batch_size']))
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

batch_data = iterator.get_next()

model = CompletionModel()
g_vars, d_vars, losses = model.build_graph_with_losses(batch_data, cfg)

if cfg['val']:
    pass

# training settings
lr = tf.get_variable('lr', shape=[], trainable=False, initializer=tf.constant_initializer(1e-4))
g_opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
d_opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)

# initialize primary trainer
global_step = tf.get_variable('global_step',
                              [],
                              tf.int32,
                              initializer=tf.zeros_initializer(),
                              trainable=False)

g_loss = cfg['g_loss']
d_loss = cfg['d_loss']

g_grads_vars = g_opt.compute_gradients(g_loss, g_vars)
g_train = g_opt.apply_gradients(g_grads_vars, global_step)

d_grads_vars = d_opt.compute_gradients(d_loss, d_vars)
d_train = d_opt.apply_gradients(d_grads_vars, global_step)

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

    step = 0
    max_iters = cfg['max_iters']
    while step < max_iters:
        _, loss_value = sess.run([g_train, losses['g_loss']])
