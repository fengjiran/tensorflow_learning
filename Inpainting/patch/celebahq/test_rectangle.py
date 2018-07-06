from __future__ import print_function

import os
import platform
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
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\test_generative_inpainting\\model_logs\\release_celebahq_256'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        prefix = '/home/icie/Datasets/celebahq'
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/test_generative_inpainting/model_logs/release_celebahq_256'

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

# saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # sess.run(iterator.initializer, feed_dict={filenames: tfrecord_filenames})
