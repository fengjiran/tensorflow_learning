"""TF Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
