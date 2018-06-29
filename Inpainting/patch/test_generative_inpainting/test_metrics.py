import os
import platform
# import yaml
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


def parse_tfrecord(example_proto):
    features = {'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.uint8)
    # img = tf.reshape(data, parsed_features['shape'])
    img = tf.reshape(data, [1024, 1024, 3])

    return img


if platform.system() == 'Windows':
    val_path = 'F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001'
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\test_generative_inpainting\\model_logs\\release_celebahq_256'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/test_generative_inpainting/model_logs/release_celebahq_256'


ng.get_gpus(1)
# args = parser.parse_args()

model = InpaintCAModel()
