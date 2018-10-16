import os
import scipy.misc
import numpy as np
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist


class ImageData:

    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img
