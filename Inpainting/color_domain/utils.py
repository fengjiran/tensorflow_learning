import os
import sys
import time
import numpy as np
from scipy.misc import imread
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


def create_mask(height, width, mask_height, mask_width, x=None, y=None):
    top = y if y is not None else tf.random_uniform([], minval=0, maxval=height - mask_height, dtype=tf.int32)
    left = x if x is not None else tf.random_uniform([], minval=0, maxval=width - mask_width, dtype=tf.int32)

    mask = tf.pad(tensor=tf.ones((mask_height, mask_width), dtype=tf.float32),
                  paddings=[[top, height - mask_height - top],
                            [left, width - mask_width - left]])

    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    return mask  # [1, height, width, 1]


def images_summary(images, name, max_outs):
    """Summary images.

    **Note** that images should be scaled to [-1, 1] for 'RGB' or 'BGR',
    [0, 1] for 'GREY'.
    :param images: images tensor (in NHWC format)
    :param name: name of images summary
    :param max_outs: max_outputs for images summary
    :param color_format: 'BGR', 'RGB' or 'GREY'
    :return: None
    """
    img = (images + 1) / 2.
    tf.summary.image(name, img, max_outs)


def canny_wrap(image, sigma, mask, use_quantiles=False, low_threshold=None, high_threshold=None):
    return canny(image, sigma, low_threshold, high_threshold, mask, use_quantiles)


def tf_canny(image, sigma, mask):
    edge = tf.py_func(func=canny_wrap,
                      inp=[image, sigma, mask],
                      Tout=tf.bool)
    return edge


def get_color_domain(img, blur_factor1, blur_factor2, k):  # img:[0, 255]
    img_blur = cv2.medianBlur(img, blur_factor1)
    Z = img_blur.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 8
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img_blur.shape))

    img_color_domain = cv2.medianBlur(res, blur_factor2)

    img_color_domain = img_color_domain / 255.
    img_color_domain = img_color_domain.astype(np.float32)
    return img_color_domain  # [0, 1]


def tf_get_color_domain(img, blur_factor1, blur_factor2, k):
    img_color_domain = tf.py_func(func=get_color_domain,
                                  inp=[img, blur_factor1, blur_factor2, k],
                                  Tout=tf.float32)
    return img_color_domain


if __name__ == '__main__':
    img = tf.placeholder(tf.uint8, [1024, 1024, 3])
    img_color_domain = tf_get_color_domain(img, 21, 3, 3)

    img1 = imread('img.png')

    with tf.Session() as sess:
        res = sess.run(img_color_domain, feed_dict={img: img1})
        print(res.shape)

        plt.figure()

        plt.subplot(121)
        plt.imshow(img1)
        plt.axis('off')
        plt.title('rgb', fontsize=20)

        plt.subplot(122)
        plt.imshow(res)
        plt.axis('off')
        plt.title('color_domain', fontsize=20)

        plt.show()
