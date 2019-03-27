import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import tensorflow as tf


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def rgb_to_lab(srgb):  # srgb in [0, 1]
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + \
                (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4 / 29) * \
                linear_mask + (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


if __name__ == '__main__':
    img = imread('img.png')
    print(img.shape)

    a = tf.placeholder(tf.float32, [1024, 1024, 3])
    lab = rgb_to_lab(a)

    with tf.Session() as sess:
        b = sess.run(lab, feed_dict={a: img / 255.})
        l_comp = b[:, :, 0]
        a_comp = b[:, :, 1]
        b_comp = b[:, :, 2]

        l_comp /= 100.
        a_comp = (a_comp + 128) / 255.
        b_comp = (b_comp + 128) / 255.
        # print(b.shape)
        # print(b[:, :, 0].min(), b[:, :, 0].max())
        # print(b[:, :, 1].min(), b[:, :, 1].max())
        # print(b[:, :, 2].min(), b[:, :, 2].max())

        plt.figure()

        plt.subplot(141)
        plt.imshow(img)
        plt.title('rgb')

        plt.subplot(142)
        plt.imshow(l_comp, cmap=plt.cm.gray)
        plt.title('l_comp')

        plt.subplot(143)
        plt.imshow(a_comp, cmap=plt.cm.gray)
        plt.title('a_comp')

        plt.subplot(144)
        plt.imshow(b_comp, cmap=plt.cm.gray)
        plt.title('b_comp')

        # plt.imshow((b[:, :, 1] + 128) / 255., cmap=plt.cm.gray)
        plt.show()
