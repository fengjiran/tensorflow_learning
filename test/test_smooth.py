import numpy as np
import tensorflow as tf


def smooth(x, sigma):
    channel = x.get_shape().as_list()[-1]
    size_denom = 5.
    sigma = int(sigma * size_denom)
    kernel_size = sigma
    mgrid = tf.range(start=0, limit=kernel_size, dtype=tf.float32)
    mean = (kernel_size - 1.) / 2
    mgrid = mgrid - mean
    mgrid = mgrid * size_denom
    kernel = 1. / (sigma * tf.sqrt(2 * np.pi)) * tf.exp(-0.5 * tf.square(mgrid / sigma))

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / tf.reduce_sum(kernel)

    # Reshape to depthwise convolutional weight
    kernelx = tf.reshape(kernel, [int(kernel_size), 1, 1, 1])
    kernelx = tf.tile(kernelx, [1, 1, channel, channel])
    kernely = tf.reshape(kernel, [1, int(kernel_size), 1, 1])
    kernely = tf.tile(kernely, [1, 1, channel, channel])

    pad = kernel_size // 2
    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

    x = tf.nn.conv2d(x, filter=kernelx, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.conv2d(x, filter=kernely, strides=[1, 1, 1, 1], padding='VALID')

    return x


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [10, 112, 112, 3])
    x = smooth(x, sigma=1)
    print(x.get_shape())
