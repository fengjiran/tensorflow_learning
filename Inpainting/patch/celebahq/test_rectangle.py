from __future__ import print_function

import os
import platform
import numpy as np
# import yaml
import tensorflow as tf
from model import CompletionModel


def bbox2mask_np(bbox, height, width):
    top, left, h, w = bbox
    mask = np.pad(array=np.ones((h, w)),
                  pad_width=((top, height - h - top), (left, width - w - left)),
                  mode='constant',
                  constant_values=0)
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, -1)
    mask = np.concatenate((mask, mask, mask), axis=3) * 255
    return mask


if platform.system() == 'Windows':
    prefix = 'F:\\Datasets\\celebahq'
    val_path = 'F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001'
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\celebahq\\model\\refine'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        prefix = '/home/icie/Datasets/celebahq'
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/celebahq/model/refine'

hole_size = 128
image_size = 256
bbox_np = ((image_size - hole_size) // 2,
           (image_size - hole_size) // 2,
           hole_size,
           hole_size)
mask = bbox2mask_np(bbox_np, image_size, image_size)

model = CompletionModel()
image_ph = tf.placeholder(tf.float32, (1, 256, 256, 3))
mask_ph = tf.placeholder(tf.float32, (1, 256, 256, 3))
inputs = tf.concat([image_ph, mask_ph], axis=2)
batch_incomplete, batch_complete_coarse, batch_complete_refine = model.build_test_graph(inputs)
batch_complete_refine = (batch_complete_refine + 1.) * 127.5
batch_complete_refine = tf.saturate_cast(batch_complete_refine, tf.uint8)

# metrics
# image value in (0,255)
ssim_tf = tf.image.ssim(tf.cast(image_ph[0], tf.uint8), batch_complete_refine[0], 255)
psnr_tf = tf.image.psnr(tf.cast(image_ph[0], tf.uint8), batch_complete_refine[0], 255)
tv_loss = tf.image.total_variation(image_ph[0]) -\
    tf.image.total_variation(tf.cast(batch_complete_refine[0], dtype=tf.float32))
tv_loss = tv_loss / tf.image.total_variation(image_ph[0])

# image value in (-1,1)
l1_loss = tf.reduce_mean(tf.abs(image_ph[0] -
                                tf.cast(batch_complete_refine[0], dtype=tf.float32))) / 127.5
l2_loss = tf.reduce_mean(tf.square(image_ph[0] -
                                   tf.cast(batch_complete_refine[0], dtype=tf.float32))) / 16256.25

ssims = []
psnrs = []
l1_losses = []
l2_losses = []
tv_losses = []

# saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    pass
