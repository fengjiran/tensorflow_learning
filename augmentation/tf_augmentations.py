"""TF Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    pi = tf.constant(math.pi)
    quadrant = tf.bitwise.bitwise_and(tf.cast(tf.math.floor(angle / (pi / 2)), dtype=tf.int32), 3)
    sign_alpha = tf.cond(tf.equal(tf.bitwise.bitwise_and(quadrant, 1), 0), lambda: angle, lambda: pi - angle)
    alpha = tf.math.floormod(tf.math.floormod(sign_alpha, pi) + pi, pi)
    bb_w = w * tf.math.cos(alpha) + h * tf.math.sin(alpha)
    bb_h = w * tf.math.sin(alpha) + h * tf.math.cos(alpha)
    gamma = tf.cond(tf.less(w, h), lambda: tf.math.atan2(bb_w, bb_w), lambda: tf.math.atan2(bb_w, bb_w))
    delta = pi - alpha - gamma
    length = tf.cond(tf.less(w, h), lambda: h, lambda: w)
    d = length * tf.math.cos(alpha)
    a = d * tf.math.sin(alpha) / tf.math.sin(delta)

    y = a * tf.math.cos(gamma)
    x = y * tf.math.tan(gamma)

    return bb_w - 2 * x, bb_h - 2 * y
    

def random_flip(img):
    """
    Flip the input x horizontally with 50% probability.

    When passing a batch of images, each image will be randomly flipped independent of other images.
    """
    img = tf.image.random_flip_left_right(img)
    return img


# def random_flip(x):
#     """Flip the input x horizontally with 50% probability."""
#     x = tf.cond(tf.random.uniform([]) > 0.5, lambda: tf.image.flip_left_right(x), lambda: tf.identity(x))
#     return x


def zero_pad_and_crop(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.

    Args
    ----
      img: tf image that will be zero padded and cropped.
      amount: amount of zeros to pad `img` with horizontally and verically.

    Returns
    -------
      The cropped zero padded img. The returned tf image will be of the same
      shape as `img`.

    """
    shape = img.get_shape()
    assert len(shape) == 4
    padded_img = tf.pad(img, paddings=tf.constant([[0, 0], [amount, amount], [amount, amount], [0, 0]]))
    padded_and_cropped_img = [tf.random_crop(a, shape[1:]) for a in tf.unstack(padded_img, axis=0)]
    padded_and_cropped_img = tf.stack(padded_and_cropped_img)

    # uniform_random = tf.random.uniform([batch_size], 0, 1.0)
    # crops = tf.round(tf.reshape(uniform_random, [batch_size, 1, 1, 1]))
    # crops = tf.cast(crops, img.dtype)
    # cropped_img = tf.random_crop(padded_img, shape)
    # return crops * cropped_img + (1 - crops) * img

    return padded_and_cropped_img


def cutout(img, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.

    The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
    This operation applies a `size`x`size` mask of zeros to a random location
    within `img`.

    Args:
    ----
      img: tf tensor that cutout will be applied to.
      size: Height/width of the cutout mask that will be

    Returns
    -------
      A tf tensor that is the result of applying the cutout mask to `img`.

    """


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Create a zero mask used for cutout of shape `img_height` x `img_width`.

    Args:
    ----
      img_height: Height of image cutout mask will be applied to.
      img_width: Width of image cutout mask will be applied to.
      num_channels: Number of channels in the image.
      size: Size of the zeros mask.

    Returns
    -------
      A mask of shape `img_height` x `img_width` with all ones except for a
      square of zeros of shape `size` x `size`. This mask is meant to be
      elementwise multiplied with the original image. Additionally returns
      the `upper_coord` and `lower_coord` which specify where the cutout mask
      will be applied.

    """
    # Sample center where cutout mask will be applied
    height_loc = tf.random.uniform([], minval=0, maxval=img_height, dtype=tf.int32)
    width_loc = tf.random.uniform([], minval=0, maxval=img_width, dtype=tf.int32)

    # Determine upper right and lower left corners of patch
    upper_coord = (tf.maximum(0, height_loc - size // 2), tf.maximum(0, width_loc - size // 2))
    lower_coord = (tf.minimum(img_height, height_loc + size // 2), tf.minimum(img_width, width_loc + size // 2))

    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]

    mask = tf.ones((img_height, img_width, num_channels))
    zeros = tf.zeros((mask_height, mask_width, num_channels))

    # mask = mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :].assign(zeros)
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = zeros
    return mask


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
    ----
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns
    -------
      A blended image Tensor of type uint8.

    """

    def f1():
        img = image1
        return img

    def f2():
        return image2

    def f3():
        # img1 = image1
        # img2 = image2
        img1 = tf.to_float(image1)
        img2 = tf.to_float(image2)

        difference = img2 - img1
        scaled = factor * difference

        temp = img1 + scaled


if __name__ == '__main__':
    import numpy as np
    import imageio
    import matplotlib.pyplot as plt

    img = imageio.imread('quokka.jpg')
    print(type(img))
    print(img.shape)
    w, h, c = img.shape
    N = 3

    img1 = np.reshape(img, [1, w, h, c])
    img2 = np.reshape(img, [1, w, h, c])
    img3 = np.reshape(img, [1, w, h, c])

    img_tf = tf.placeholder(tf.float32, [N, w, h, c])

    tmp = random_flip(img_tf)
    tmp = zero_pad_and_crop(tmp, 200)

    mask = create_cutout_mask(32, 32, 3, 16)

    with tf.Session() as sess:
        img_res = sess.run(tmp, feed_dict={img_tf: np.concatenate([img1, img2, img3], axis=0)})
        # print(img_res.max())
        # print(img_res.min())

    img_res1 = np.reshape(img_res[0], [w, h, c])
    img_res2 = np.reshape(img_res[1], [w, h, c])
    img_res3 = np.reshape(img_res[2], [w, h, c])

    plt.figure()
    plt.subplot(131)
    plt.imshow(img_res1 / 255)
    plt.subplot(132)
    plt.imshow(img_res2 / 255)
    plt.subplot(133)
    plt.imshow(img_res3 / 255)
    plt.show()
