# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.

Usage:
python msssim.py --original_image=original.png --compared_image=distorted.png

"""
from __future__ import division
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf


def _FSpecialGauss(size, sigma):
    """Construct function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1

    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _tf_fspecial_gaussian(size, sigma):
    """Construct function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1

    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=-1)

    y = np.expand_dims(y, axis=-1)
    y = np.expand_dims(y, axis=-1)

    # assert len(x) == size
    g = tf.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5, multichannel=False):
    if multichannel:
        nch = img1.get_shape()[-1]
        value = np.zeros(nch)
        for ch in range(nch):
            ch_result = tf_ssim(img1[:, :, :, ch], img2[:, :, :, ch])

    window = _tf_fspecial_gaussian(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 255  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).

    Return:
    -------
        Pair containing the mean SSIM and contrast sensitivity between `img1` and
        `img2`.

    Raise:
    ------
        RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].

    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)

    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode="valid")
        mu2 = signal.fftconvolve(img2, window, mode="valid")

        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)

    return ssim, cs


if __name__ == '__main__':
    w1 = _FSpecialGauss(size=11, sigma=1.5)
    # print(w)
    # print(w.sum())
    print(w1.shape)

    w2 = _tf_fspecial_gaussian(11, 1.5)

    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     print(sess.run(w2))

    # print(w2.get_shape())
