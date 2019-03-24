import tensorflow as tf

# weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# weight_regularizer = None
#


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
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
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

        x_shape = x.get_shape().as_list()
        output_shape = [-1, x_shape[1] * stride, x_shape[2] * stride, channels]

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


def attention(x, ch, sn=False, scope='attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        pass
