import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_mask(height, width, mask_height, mask_width, x=None, y=None):
    top = tf.random_uniform([], minval=0, maxval=height - mask_height, dtype=tf.int32)
    left = tf.random_uniform([], minval=0, maxval=width - mask_width, dtype=tf.int32)

    mask = np.zeros([height, width])
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1.
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap *
                            (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            pass
