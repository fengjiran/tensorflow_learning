import os
import cv2
import numpy as np


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
