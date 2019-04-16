import os
import yaml
import platform as pf
import numpy as np
import tensorflow as tf
import cv2
from .networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    checkpoint_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7610']


model = ColorModel(cfg)
