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

gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
