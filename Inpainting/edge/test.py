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

edge_model = EdgeModel(cfg)

# 1 for missing region, 0 for background
mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])

gray_masked = gray * (1 - mask) + mask
edge_masked = edge * (1 - mask)

inputs = tf.concat([gray_masked, edge_masked, mask * tf.ones_like(gray)], axis=3)

outputs = edge_model.edge_generator(inputs)
