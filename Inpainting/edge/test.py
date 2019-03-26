from __future__ import print_function

import os
import yaml
import platform as pf
import numpy as np
# import cv2
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import tensorflow as tf
from networks import EdgeModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    checkpoint_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass


edge_model = EdgeModel(cfg)

# 1 for missing region, 0 for background
mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])

gray_masked = gray * (1 - mask) + mask
edge_masked = edge * (1 - mask)

inputs = tf.concat([gray_masked, edge_masked, mask * tf.ones_like(gray)], axis=3)

outputs = edge_model.edge_generator(inputs)
outputs_merged = outputs * mask + edge * (1 - mask)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'edge_generator')
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.train.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
