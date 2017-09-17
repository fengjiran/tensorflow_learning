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
# from scipy.ndimage.filters import convolve
from skimage import io
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

    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)

    # assert len(x) == size
    g = tf.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5, multichannel=False):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    """
    if multichannel:
        nch = img1.get_shape()[-1]

        if cs_map:
            ssim = []
            cs = []
            for ch in range(nch):
                img1_ch = img1[:, :, :, ch]
                img2_ch = img2[:, :, :, ch]

                img1_ch = tf.expand_dims(img1_ch, axis=-1)
                img2_ch = tf.expand_dims(img2_ch, axis=-1)

                ssim_map, cross_scale_map = tf_ssim(img1_ch, img2_ch, cs_map=cs_map,
                                                    mean_metric=False, multichannel=False)
                ssim.append(ssim_map)
                cs.append(cross_scale_map)

            n = len(ssim)
            ssim_map = tf.add_n(ssim) / n
            cross_scale_map = tf.add_n(cs) / n

            return ssim_map, cross_scale_map
        else:
            value = []
            for ch in range(nch):
                img1_ch = img1[:, :, :, ch]
                img2_ch = img2[:, :, :, ch]

                img1_ch = tf.expand_dims(img1_ch, axis=-1)
                img2_ch = tf.expand_dims(img2_ch, axis=-1)

                ch_result = tf_ssim(img1_ch, img2_ch, cs_map=cs_map, multichannel=False)
                value.append(ch_result)

            return tf.reduce_mean(value)

    window = _tf_fspecial_gaussian(size, sigma)  # window shape [size, size, 1, 1]
    # window = tf.cast(window, tf.float32)
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

        if mean_metric:
            return (tf.reduce_mean((1.0 - value[0]) / 2), tf.reduce_mean((1.0 - value[1]) / 2))
        else:
            return ((1.0 - value[0]) / 2, (1.0 - value[1]) / 2)
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            return tf.reduce_mean((1.0 - value) / 2)
        else:
            return (1.0 - value) / 2


def tf_ms_ssim(img1, img2, mean_metric=True, level=5, multichannel=False):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    """
    weights = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, multichannel=multichannel)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))

        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        img1 = filtered_im1
        img2 = filtered_im2

    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:level - 1]**weights[0:level - 1]) * (mssim[level - 1]**weights[level - 1])

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_l1_loss(img1, img2, size=11, sigma=1.5):
    """Compute l1 loss with gaussian window."""
    diff = tf.abs(img1 - img2)
    window = _tf_fspecial_gaussian(size, sigma)
    tensor_shape = img1.get_shape().as_list()
    assert len(tensor_shape) == 4

    nch = tensor_shape[-1]

    if nch > 1:  # multi channel
        results = []
        for ch in range(nch):
            diff_ch = diff[:, :, :, ch]
            diff_ch = tf.expand_dims(diff_ch, axis=-1)
            result_ch = tf.nn.conv2d(diff_ch, window, [1, 1, 1, 1], padding='VALID')
            results.append(result_ch)

        l1_loss = tf.reduce_mean(tf.add_n(results))
    else:  # single channel
        result = tf.nn.conv2d(diff, window, [1, 2, 2, 1], padding='VALID')
        l1_loss = tf.resuce_mean(result)

    return l1_loss


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
    # w1 = _FSpecialGauss(size=11, sigma=1.5)
    # print(w)
    # print(w.sum())
    # print(w1.shape)

    # w2 = _tf_fspecial_gaussian(11, 1.5)
    img1 = io.imread('C:\\Users\\Richard\\Desktop\\jie.jpg').astype(np.float32)
    img2 = io.imread('C:\\Users\\Richard\\Desktop\\jie.jpg').astype(np.float32)

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    x = tf.placeholder(tf.float32, img1.shape)
    y = tf.placeholder(tf.float32, img2.shape)

    # result = tf_ms_ssim(x, y, multichannel=True)
    result = tf_l1_loss(x, y)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        print(sess.run(result, feed_dict={x: img1, y: img2}))
