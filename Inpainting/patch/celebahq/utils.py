from __future__ import print_function

import yaml
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import ops


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimentions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1.0) / 2.0


def rgb2lab(srgb):
    srgb = check_image(srgb)
    srgb_pixels = tf.reshape(srgb, [-1, 3])

    # srgb to xyz
    linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
    exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055)**2.4) * exponential_mask

    rgb2xyz = tf.constant([
        [0.412453, 0.212671, 0.019334],
        [0.357580, 0.715160, 0.119193],
        [0.180423, 0.072169, 0.950227]
    ])

    xyz_pixels = tf.matmul(rgb_pixels, rgb2xyz)

    xyz_normalized_pixels = tf.multiply(xyz_pixels, [1.0 / 0.950456, 1.0, 1.0 / 1.088754])

    epsilon = 6.0 / 29.0
    linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
    exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3.0 * epsilon**2) + 4.0 / 29.0) * \
        linear_mask + (xyz_normalized_pixels ** (1.0 / 3.0)) * exponential_mask


def spatial_discounting_mask(cfg):
    gamma = cfg['spatial_discount_gamma']
    height = cfg['hole_height']
    width = cfg['hole_width']
    shape = [1, height, width, 1]

    if cfg['discount_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = gamma**min(i, j, height - i, width - j)

        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
    else:
        mask_values = np.ones(shape)

    return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def random_bbox(cfg):
    # image_shape:(H,W,C)
    height = cfg['img_height']
    width = cfg['img_width']

    hole_height = cfg['hole_height']
    hole_width = cfg['hole_width']

    bbox = []

    for _ in range(cfg['batch_size']):
        top = tf.random_uniform([], minval=0, maxval=height - hole_height, dtype=tf.int32)
        left = tf.random_uniform([], minval=0, maxval=width - hole_width, dtype=tf.int32)
        h = tf.constant(hole_height)
        w = tf.constant(hole_width)
        bbox.append((top, left, h, w))

    return bbox


def bbox2mask(bbox, cfg):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns
    -------
        tf.Tensor: output with shape [bs, H, W, 1]

    """
    height = cfg['img_height']
    width = cfg['img_width']

    masks = []

    for (top, left, h, w) in bbox:
        mask = tf.pad(tensor=tf.ones((h, w), dtype=tf.float32),
                      paddings=[[top, height - h - top],
                                [left, width - w - left]])

        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)

        masks.append(mask)

    return tf.concat(masks, axis=0)


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns
    -------
        tf.Tensor: local patch

    """
    patches = []
    batch_size = x.get_shape().as_list()[0]
    assert batch_size == len(bbox)
    for i in range(batch_size):
        patch = tf.image.crop_to_bounding_box(x[i], bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3])
        patch = tf.expand_dims(patch, 0)
        patches.append(patch)

    # x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return tf.concat(patches, axis=0)


def gan_wgan_loss(pos, neg):
    g_loss = -tf.reduce_mean(neg)
    d_loss = tf.reduce_mean(neg - pos)

    return g_loss, d_loss


def random_interpolates(x, y, alpha=None):
    """Generate.

    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])

    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha * (y - x)
    return tf.reshape(interpolates, shape)


def gradient_penalty(x, y, mask=None, norm=1.):
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm) / (norm**2))


def lipschitz_penalty(x, y, mask=None, norm=1.):
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(tf.nn.relu(slopes - norm)))


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
    # img = tf.cast((images + 1) * 127.5, tf.int8)
    img = (images + 1) / 2.
    tf.summary.image(name, img, max_outs)
    # with tf.variable_scope(name), tf.device('/cpu:0'):
    #     if color_format == 'BGR':
    #         img = tf.clip_by_value(
    #             (tf.reverse(images, [-1]) + 1.) * 127.5, 0., 255.)
    #     elif color_format == 'RGB':
    #         # img = tf.clip_by_value((images + 1.) * 127.5, 0, 255)
    #         # img = (images + 1) / 2
    #         img = tf.cast((img + 1) * 127.5, tf.int8)
    #     elif color_format == 'GREY':
    #         img = tf.clip_by_value(images * 255., 0, 255)
    #     else:
    #         raise NotImplementedError("color format is not supported.")
    #     tf.summary.image(name, img, max_outputs=max_outs)


def gradients_summary(y, x, norm=tf.abs, name='gradients_y_wrt_x'):
    grad = tf.reduce_mean(norm(tf.gradients(y, x)))
    tf.summary.scalar(name, grad)


def instance_norm(x, name="instance_norm"):
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable("scale", [depth],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth],
                                 initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset


# weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_init = tf.contrib.layers.xavier_initializer_conv2d()
weight_regularizer = None


def atrous_conv(x, channels, kernel=3, dilation=1, use_bias=True, sn=True, name='conv_0'):
    with tf.variable_scope(name):
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

            x = tf.nn.atrous_conv2d(value=x,
                                    filters=spectral_norm(w),
                                    rate=dilation,
                                    padding='SAME')
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 use_bias=use_bias, dilation_rate=dilation)

        return x


def conv(x, channels, kernel=4, stride=1, dilation=1,
         pad=0, pad_type='zero', use_bias=True, sn=True, name='conv_0'):
    with tf.variable_scope(name):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x,
                             filter=spectral_norm(w),
                             strides=[1, stride, stride, 1],
                             dilations=[1, dilation, dilation, 1],
                             padding='VALID',
                             data_format='NHWC')

            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=1, use_bias=True, sn=True, name='deconv_0'):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w),
                                       output_shape=output_shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME')

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel,
                                           kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride,
                                           padding='SAME',
                                           use_bias=use_bias)

        return x


def resnet_block(x, out_channels, dilation=1, name='resnet_block'):
    with tf.variable_scope(name):
        y = atrous_conv(x, out_channels, kernel=3, dilation=dilation, name='conv1')
        # y = conv(x, out_channels, kernel=3, stride=1, dilation=dilation,
        #          pad=dilation, pad_type='reflect', name='conv1')
        y = instance_norm(y, name='in1')
        y = tf.nn.relu(y)

        y = conv(y, out_channels, kernel=3, stride=1, dilation=1,
                 pad=1, pad_type='reflect', name='conv2')
        y = instance_norm(y, name='in2')

        return x + y


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)

    u_hat = u
    v_hat = None

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    x = tf.random_uniform([cfg['batch_size'], 256, 256, 3])
    bbox = random_bbox(cfg)
    patches = local_patch(x, bbox)
    mask = bbox2mask(bbox, cfg)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        bbox, mask, patches = sess.run([bbox, mask, patches])
        print(bbox)
        print(mask.shape)
        print(patches.shape)
