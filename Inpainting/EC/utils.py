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
    top = y if y is not None else tf.random_uniform([], minval=0, maxval=height - mask_height, dtype=tf.int32)
    left = x if x is not None else tf.random_uniform([], minval=0, maxval=width - mask_width, dtype=tf.int32)

    mask = tf.pad(tensor=tf.ones((mask_height, mask_width), dtype=tf.float32),
                  paddings=[[top, height - mask_height - top],
                            [left, width - mask_width - left]])

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
