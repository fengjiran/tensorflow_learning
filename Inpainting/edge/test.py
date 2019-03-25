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
