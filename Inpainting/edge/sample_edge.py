import os
import yaml
import platform as pf
import numpy as np
import cv2
from scipy.misc import imread
from scipy.misc import imsave
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray
import tensorflow as tf

from networks import EdgeModel


def load_mask(cfg, mask_type=1, mask_path=None):
    if mask_type == 1:  # random block
        hole_size = cfg['INPUT_SIZE'] // 2
        top = np.random.randint(0, hole_size + 1)
        left = np.random.randint(0, hole_size + 1)
        img_mask = np.pad(array=np.ones((hole_size, hole_size)),
                          pad_width=((top, cfg['INPUT_SIZE'] - hole_size - top),
                                     (left, cfg['INPUT_SIZE'] - hole_size - left)),
                          mode='constant',
                          constant_values=0)

        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)  # (1, 256, 256, 1) float
    else:  # external
        img_mask = imread(mask_path)
        img_mask = cv2.resize(img_mask, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)
        img_mask = img_mask > 3
        img_mask = img_mask.astype(np.float32)
        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)
        img_mask = 1 - img_mask

    return img_mask  # (1, 256, 256, 1)


def load_edge(cfg, image_path):
    image = imread(image_path)  # [1024, 1024, 3], [0, 255]
    image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
    gray = rgb2gray(image)  # [256, 256], [0, 1]

    edge = canny(gray, sigma=cfg['SIGMA'])
    edge = edge.astype(np.float32)

    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    edge = np.expand_dims(edge, 0)
    edge = np.expand_dims(edge, -1)

    return gray, edge
