import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import tensorflow as tf


def conv(x, channels, kernel=4, stride=1, dilation=1,
         pad=0, pad_type='zero', use_bias=True, sn=True, init_type='normal', name='conv_0'):
    with tf.variable_scope(name):
        if init_type == 'normal':
            weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        elif init_type == 'xavier':
            weight_init = tf.glorot_normal_initializer()
        elif init_type == 'kaiming':
            weight_init = tf.keras.initializers.he_normal()
        elif init_type == 'orthogonal':
            weight_init = tf.orthogonal_initializer(gain=0.02)

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, tf.shape(x)[-1], channels],
                                initializer=weight_init,
                                regularizer=None)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x,
                             filter=spectral_norm(w),
                             strides=[1, stride, stride, 1],
                             dilations=[1, dilation, dilation, 1],
                             padding='VALID')

            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=None,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=1, use_bias=True, sn=True, init_type='normal', name='deconv_0'):
    with tf.variable_scope(name):
        if init_type == 'normal':
            weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        elif init_type == 'xavier':
            weight_init = tf.glorot_normal_initializer()
        elif init_type == 'kaiming':
            weight_init = tf.keras.initializers.he_normal()
        elif init_type == 'orthogonal':
            weight_init = tf.orthogonal_initializer(gain=0.02)

        # x_shape = x.get_shape().as_list()
        x_shape = tf.shape(x)
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]],
                                initializer=weight_init,
                                regularizer=None)
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
                                           kernel_regularizer=None,
                                           strides=stride,
                                           padding='SAME',
                                           use_bias=use_bias)

        return x


def atrous_conv(x, channels, kernel=3, dilation=1, use_bias=True, sn=True, init_type='normal', name='conv_0'):
    with tf.variable_scope(name):
        if init_type == 'normal':
            weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        elif init_type == 'xavier':
            weight_init = tf.glorot_normal_initializer()
        elif init_type == 'kaiming':
            weight_init = tf.keras.initializers.he_normal()
        elif init_type == 'orthogonal':
            weight_init = tf.orthogonal_initializer(gain=0.02)

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=weight_init,
                                regularizer=None)
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
                                 kernel_regularizer=None,
                                 use_bias=use_bias, dilation_rate=dilation)

        return x


def resnet_block(x, out_channels, dilation=1, init_type='normal', sn=True, name='resnet_block'):
    with tf.variable_scope(name):
        y = atrous_conv(x, out_channels, kernel=3, dilation=dilation, sn=sn,
                        init_type=init_type, name='conv1')
        # y = conv(x, out_channels, kernel=3, stride=1, dilation=dilation,
        #          pad=dilation, pad_type='reflect', name='conv1')
        y = instance_norm(y, name='in1')
        y = tf.nn.relu(y)

        y = conv(y, out_channels, kernel=3, stride=1, dilation=1, sn=sn,
                 pad=1, pad_type='reflect', init_type=init_type, name='conv2')
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
        plt.axis('off')
        plt.title('rgb')

        plt.subplot(142)
        plt.imshow(l_comp, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('l_comp')

        plt.subplot(143)
        plt.imshow(a_comp, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('a_comp')

        plt.subplot(144)
        plt.imshow(b_comp, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('b_comp')

        # plt.imshow((b[:, :, 1] + 128) / 255., cmap=plt.cm.gray)
        plt.show()
