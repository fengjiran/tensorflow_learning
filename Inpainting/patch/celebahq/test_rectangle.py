from __future__ import print_function

import os
import platform
import yaml
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


def parse_tfrecord(example_proto):
    features = {'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.uint8)
    # img = tf.reshape(data, parsed_features['shape'])
    img = tf.reshape(data, [1024, 1024, 3])

    return img


val_filenames = tf.placeholder(tf.string, shape=[None])
val_data = tf.data.TFRecordDataset(val_filenames)
val_data = val_data.map(parse_tfrecord)
# val_data = val_data.batch(1000)
# val_data = val_data.repeat()
val_iterator = val_data.make_initializable_iterator()
val_batch_data = val_iterator.get_next()
val_batch_data = tf.image.resize_area(val_batch_data, [256, 256])
val_batch_data = tf.clip_by_value(val_batch_data, 0., 255.)
val_batch_data = val_batch_data / 127.5 - 1

for _, _, files in os.walk(val_path):
    val_tfrecord_filenames = files
val_tfrecord_filenames = [os.path.join(val_path, file) for file in val_tfrecord_filenames]


model = CompletionModel()
g_vars, d_vars, losses = model.build_graph_with_losses(val_batch_data, cfg)
batch_incomplete, batch_complete_coarse, batch_complete_refine = model.test(val_batch_data, cfg)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # sess.run(iterator.initializer, feed_dict={filenames: tfrecord_filenames})
    sess.run(val_iterator.initializer, feed_dict={val_filenames: val_tfrecord_filenames})
    saver.restore(sess, os.path.join(refine_model_path, 'refine_model'))

    val_l1_loss = tf.reduce_mean(tf.abs(val_batch_data - batch_complete_refine))
    val_l2_loss = tf.reduce_mean(tf.square(val_batch_data - batch_complete_refine))
    psnr = tf.reduce_mean(tf.image.psnr(val_batch_data, batch_complete_refine, 2))
    ssim = tf.reduce_mean(tf.image.ssim(val_batch_data, batch_complete_refine, 2))

    m1, m2, m3, m4 = sess.run([val_l1_loss, val_l2_loss, psnr, ssim])
