import tensorflow as tf

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


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
                             padding='VALID')

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
        y = conv(x, out_channels, kernel=3, stride=1, dilation=dilation,
                 pad=dilation, pad_type='reflect', name='conv1')
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
