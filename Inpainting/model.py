from __future__ import print_function

import numpy as np
import pickle
import tensorflow as tf


def conv_layer(inputs, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=filter_shape[-1],
                            initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(inputs, w, [1, stride, stride, 1], padding=padding)

    return activation(tf.nn.bias_add(conv, b))


def deconv_layer(inputs, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=filter_shape[-2],
                            initializer=tf.constant_initializer(0.))

        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)
    return activation(tf.nn.bias_add(deconv, b))


def fc_layer(inputs, output_size, name, activation=tf.identity):
    shape = inputs.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(inputs, [-1, dim])   # flatten
    input_size = dim

    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[output_size],
                            initializer=tf.constant_initializer(0.))

    return activation(tf.nn.bias_add(tf.matmul(x, w), b))


def channel_wise_fc_layer(inputs, name, activation=tf.identity):  # bottom:(7,7,512)
    _, width, height, n_feat_map = inputs.get_shape().as_list()
    inputs_reshape = tf.reshape(inputs, [-1, width * height, n_feat_map])
    inputs_transpose = tf.transpose(inputs_reshape, [2, 0, 1])

    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=[n_feat_map, width * height, width * height],
                            initializer=tf.truncated_normal_initializer(0., 0.005))

        output = tf.matmul(inputs_transpose, w)

        output = tf.transpose(output, [1, 2, 0])
        output = tf.reshape(output, [-1, height, width, n_feat_map])

        # add 1*1 conv
        w_conv = tf.get_variable(name='w_conv',
                                 shape=[1, 1, n_feat_map, n_feat_map],
                                 initializer=tf.contrib.layers.xavier_initializer())

        b_conv = tf.get_variable(name='b_conv',
                                 shape=[n_feat_map],
                                 initializer=tf.constant_initializer(0.))

        output = tf.nn.conv2d(input=output,
                              filter=w_conv,
                              strides=[1, 1, 1, 1],
                              padding='SAME')

    return activation(tf.nn.bias_add(output, b_conv))


def batch_norm_layer(inputs, is_training, decay=0.999, epsilon=1e-5, name=None):
    with tf.variable_scope(name):
        scale = tf.get_variable(name='scale',
                                shape=[inputs.get_shape()[-1]],
                                initializer=tf.constant_initializer(1.))

        beta = tf.get_variable(name='beta',
                               shape=[inputs.get_shape()[-1]],
                               initializer=tf.constant_initializer(0.))

        pop_mean = tf.get_variable(name='pop_mean',
                                   shape=[inputs.get_shape()[-1]],
                                   initializer=tf.constant_initializer(0.),
                                   trainable=False)

        pop_var = tf.get_variable(name='pop_var',
                                  shape=[inputs.get_shape()[-1]],
                                  initializer=tf.constant_initializer(1.),
                                  trainable=False)

        def mean_var_update():
            axes = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axes)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, variance = tf.cond(is_training, mean_var_update, lambda: (pop_mean, pop_var))

        return tf.nn.batch_normalization(x=inputs,
                                         mean=mean,
                                         variance=variance,
                                         offset=beta,
                                         scale=scale,
                                         variance_epsilon=epsilon)

        # if is_training:
        #     axes = list(range(len(inputs.get_shape()) - 1))
        #     batch_mean, batch_var = tf.nn.moments(inputs, axes)
        #     train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        #     train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        #     with tf.control_dependencies([train_mean, train_var]):
        #         return tf.nn.batch_normalization(x=inputs,
        #                                          mean=batch_mean,
        #                                          variance=batch_var,
        #                                          offset=beta,
        #                                          scale=scale,
        #                                          variance_epsilon=epsilon)
        # else:
        #     return tf.nn.batch_normalization(x=inputs,
        #                                      mean=pop_mean,
        #                                      variance=pop_var,
        #                                      offset=beta,
        #                                      scale=scale,
        #                                      variance_epsilon=epsilon)


def reconstruction(images, is_training):
    batch_size = images.get_shape().as_list()[0]

    with tf.variable_scope('generator'):
        # encoder
        conv1 = conv_layer(images, [4, 4, 3, 64], stride=2, name='conv1')
        bn1 = batch_norm_layer(conv1, is_training, name='bn1')
        bn1 = tf.contrib.keras.layers.LeakyReLU()(bn1)

        conv2 = conv_layer(bn1, [4, 4, 64, 64], stride=2, name='conv2')
        bn2 = batch_norm_layer(conv2, is_training, name='bn2')
        bn2 = tf.contrib.keras.layers.LeakyReLU()(bn2)

        conv3 = conv_layer(bn2, [4, 4, 64, 128], stride=2, name='conv3')
        bn3 = batch_norm_layer(conv3, is_training, name='bn3')
        bn3 = tf.contrib.keras.layers.LeakyReLU()(bn3)

        conv4 = conv_layer(bn3, [4, 4, 128, 256], stride=2, name='conv4')
        bn4 = batch_norm_layer(conv4, is_training, name='bn4')
        bn4 = tf.contrib.keras.layers.LeakyReLU()(bn4)

        conv5 = conv_layer(bn4, [4, 4, 256, 512], stride=2, name='conv5')
        bn5 = batch_norm_layer(conv5, is_training, name='bn5')
        bn5 = tf.contrib.keras.layers.LeakyReLU()(bn5)

        conv6 = conv_layer(bn5, [4, 4, 512, 4000], stride=2, name='conv6')
        bn6 = batch_norm_layer(conv6, is_training, name='bn6')
        bn6 = tf.contrib.keras.layers.LeakyReLU()(bn6)

        # decoder
        deconv4 = deconv_layer(bn6, [4, 4, 512, 4000], conv5.get_shape().as_list(),
                               padding='VALID', stride=2, name='deconv4')
        debn4 = batch_norm_layer(deconv4, is_training, name='debn4')
        debn4 = tf.nn.relu(debn4)

        deconv3 = deconv_layer(debn4, [4, 4, 256, 512], conv4.get_shape().as_list(),
                               stride=2, name='deconv3')
        debn3 = batch_norm_layer(deconv3, is_training, name='debn3')
        debn3 = tf.nn.relu(debn3)

        deconv2 = deconv_layer(debn3, [4, 4, 128, 256], conv3.get_shape().as_list(),
                               stride=2, name='deconv2')
        debn2 = batch_norm_layer(deconv2, is_training, name='debn2')
        debn2 = tf.nn.relu(debn2)

        deconv1 = deconv_layer(debn2, [4, 4, 64, 128], conv2.get_shape().as_list(),
                               stride=2, name='deconv1')
        debn1 = batch_norm_layer(deconv1, is_training, name='debn1')
        debn1 = tf.nn.relu(debn1)

        recon = deconv_layer(debn1, [4, 4, 3, 64], [batch_size, 64, 64, 3],
                             stride=2, name='recon')

        recon = tf.nn.tanh(recon)

    return recon


def discriminator(images, is_training, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = conv_layer(images, [4, 4, 3, 64], stride=2, name='conv1')
        bn1 = batch_norm_layer(conv1, is_training, name='bn1')
        bn1 = tf.contrib.keras.layers.LeakyReLU()(bn1)

        conv2 = conv_layer(bn1, [4, 4, 64, 128], stride=2, name='conv2')
        bn2 = batch_norm_layer(conv2, is_training, name='bn2')
        bn2 = tf.contrib.keras.layers.LeakyReLU()(bn2)

        conv3 = conv_layer(bn2, [4, 4, 128, 256], stride=2, name='conv3')
        bn3 = batch_norm_layer(conv3, is_training, name='bn3')
        bn3 = tf.contrib.keras.layers.LeakyReLU()(bn3)

        conv4 = conv_layer(bn3, [4, 4, 256, 512], stride=2, name='conv4')
        bn4 = batch_norm_layer(conv4, is_training, name='bn4')
        bn4 = tf.contrib.keras.layers.LeakyReLU()(bn4)

        output = fc_layer(bn4, output_size=1, name='output', activation=tf.nn.sigmoid)

    return output[:, 0]


if __name__ == '__main__':
    batch_size = 128
    x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='x')
    train_flag = tf.placeholder(tf.bool)
    y = reconstruction(x, train_flag)
    print(y.get_shape().as_list())

    z = discriminator(y, train_flag)
    print(z.get_shape().as_list())
