import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')

            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
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


def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def discriminator_loss(real, fake):
    real_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
    loss = real_loss + fake_loss
    return loss


def generator_loss(fake):
    loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
    return loss


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def flatten(x):
    return tf.layers.flatten(x)


def max_pooling(x, kernel=2, stride=2):
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride)


def avg_pooling(x, kernel=2, stride=2):
    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride)


def global_avg_pooling(x):
    """Incoming Tensor shape must be 4-D."""
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def spectral_norm(w, iteration=1):
    w_shape = w.shape().as_list()
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


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss
