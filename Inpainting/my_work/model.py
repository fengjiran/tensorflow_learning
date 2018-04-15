from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf


class Conv2dLayer(object):
    """Construct conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.contrib.layers.xavier_initializer())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.conv2d(self.inputs, self.w, [1, stride, stride, 1], padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))


class DeconvLayer(object):
    """Construct deconv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 output_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.contrib.layers.xavier_initializer())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-2],
                                     initializer=tf.constant_initializer(0.))

            deconv = tf.nn.conv2d_transpose(value=self.inputs,
                                            filter=self.w,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding=padding)

            self.output = activation(tf.nn.bias_add(deconv, self.b))


class FCLayer(object):
    """Construct fc layer."""

    def __init__(self,
                 inputs,
                 output_size,
                 activation=tf.identity,
                 name=None):
        self.inputs = inputs
        shape = inputs.get_shape().as_list()
        input_size = np.prod(shape[1:])
        x = tf.reshape(self.inputs, [-1, input_size])

        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=[input_size, output_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name='b',
                                     shape=[output_size],
                                     initializer=tf.constant_initializer(0.))

            self.output = activation(tf.nn.bias_add(tf.matmul(x, self.w), self.b))


class ChannelWiseLayer(object):
    """Construct channel wise layer."""

    def __init__(self, inputs, name, activation=tf.identity):
        self.inputs = inputs
        _, width, height, n_feat_map = inputs.get_shape().as_list()
        inputs_reshape = tf.reshape(inputs, [-1, width * height, n_feat_map])
        inputs_transpose = tf.transpose(inputs_reshape, [2, 0, 1])

        with tf.variable_scope(name):
            self.w_fc = tf.get_variable(name='w',
                                        shape=[n_feat_map, width * height, width * height],
                                        initializer=tf.truncated_normal_initializer(0., 0.005))
            output = tf.matmul(inputs_transpose, self.w_fc)
            output = tf.transpose(output, [1, 2, 0])
            output = tf.reshape(output, [-1, height, width, n_feat_map])

            self.w_conv = tf.get_variable(name='w_conv',
                                          shape=[1, 1, n_feat_map, n_feat_map],
                                          initializer=tf.contrib.layers.xavier_initializer())

            self.b_conv = tf.get_variable(name='b_conv',
                                          shape=[n_feat_map],
                                          initializer=tf.constant_initializer(0.))

            conv_output = tf.nn.conv2d(input=output,
                                       filter=self.w_conv,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')

            self.output = activation(tf.nn.bias_add(conv_output, self.b_conv))


def conv_layer(inputs, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=filter_shape[-1],
                            initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(inputs, w, [1, stride, stride, 1], padding=padding)
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w))

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
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w))

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
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w))

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

        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w))
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w_conv))

    return activation(tf.nn.bias_add(output, b_conv))


class BatchNormLayer(object):
    """Construct batch norm layer."""

    def __init__(self, inputs, is_training, decay=0.999, epsilon=1e-5, name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.scale = tf.get_variable(name='scale',
                                         shape=[inputs.get_shape()[-1]],
                                         initializer=tf.constant_initializer(1.))

            self.beta = tf.get_variable(name='beta',
                                        shape=[inputs.get_shape()[-1]],
                                        initializer=tf.constant_initializer(0.))

            self.pop_mean = tf.get_variable(name='pop_mean',
                                            shape=[inputs.get_shape()[-1]],
                                            initializer=tf.constant_initializer(0.),
                                            trainable=False)

            self.pop_var = tf.get_variable(name='pop_var',
                                           shape=[inputs.get_shape()[-1]],
                                           initializer=tf.constant_initializer(1.),
                                           trainable=False)

            def mean_var_update():
                axes = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs, axes)
                train_mean = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_var_update, lambda: (self.pop_mean, self.pop_var))

            self.output = tf.nn.batch_normalization(x=inputs,
                                                    mean=mean,
                                                    variance=variance,
                                                    offset=self.beta,
                                                    scale=self.scale,
                                                    variance_epsilon=epsilon)


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


def reconstruction(images, is_training):
    batch_size = images.get_shape().as_list()[0]

    with tf.variable_scope('generator'):
        # encoder
        conv1 = Conv2dLayer(images, [3, 3, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.contrib.keras.layers.LeakyReLU()(bn1_layer.output)

        tf.add_to_collection('gen_params_conv', conv1.w)
        tf.add_to_collection('gen_params_conv', conv1.b)
        tf.add_to_collection('gen_params_bn', bn1_layer.scale)
        tf.add_to_collection('gen_params_bn', bn1_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv1.w))
        print('conv1 shape:{}'.format(bn1.get_shape().as_list()))

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 64], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.contrib.keras.layers.LeakyReLU()(bn2_layer.output)

        tf.add_to_collection('gen_params_conv', conv2.w)
        tf.add_to_collection('gen_params_conv', conv2.b)
        tf.add_to_collection('gen_params_bn', bn2_layer.scale)
        tf.add_to_collection('gen_params_bn', bn2_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv2.w))
        print('conv2 shape:{}'.format(bn2.get_shape().as_list()))

        conv3 = Conv2dLayer(bn2, [3, 3, 64, 128], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.contrib.keras.layers.LeakyReLU()(bn3_layer.output)

        tf.add_to_collection('gen_params_conv', conv3.w)
        tf.add_to_collection('gen_params_conv', conv3.b)
        tf.add_to_collection('gen_params_bn', bn3_layer.scale)
        tf.add_to_collection('gen_params_bn', bn3_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv3.w))
        print('conv3 shape:{}'.format(bn3.get_shape().as_list()))

        conv4 = Conv2dLayer(bn3, [3, 3, 128, 256], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.contrib.keras.layers.LeakyReLU()(bn4_layer.output)

        tf.add_to_collection('gen_params_conv', conv4.w)
        tf.add_to_collection('gen_params_conv', conv4.b)
        tf.add_to_collection('gen_params_bn', bn4_layer.scale)
        tf.add_to_collection('gen_params_bn', bn4_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv4.w))
        print('conv4 shape:{}'.format(bn4.get_shape().as_list()))

        conv5 = Conv2dLayer(bn4, [3, 3, 256, 512], stride=2, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.contrib.keras.layers.LeakyReLU()(bn5_layer.output)

        tf.add_to_collection('gen_params_conv', conv5.w)
        tf.add_to_collection('gen_params_conv', conv5.b)
        tf.add_to_collection('gen_params_bn', bn5_layer.scale)
        tf.add_to_collection('gen_params_bn', bn5_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv5.w))
        print('conv5 shape:{}'.format(bn5.get_shape().as_list()))

        # conv6 = ChannelWiseLayer(bn5, name='cwfc')
        conv6 = Conv2dLayer(bn5, [3, 3, 512, 4000], stride=2, name='conv6')
        bn6_layer = BatchNormLayer(conv6.output, is_training, name='bn6')
        bn6 = tf.contrib.keras.layers.LeakyReLU()(bn6_layer.output)

        tf.add_to_collection('gen_params_conv', conv6.w)
        tf.add_to_collection('gen_params_conv', conv6.b)
        tf.add_to_collection('gen_params_bn', bn6_layer.scale)
        tf.add_to_collection('gen_params_bn', bn6_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv6.w))
        print('conv6 shape:{}'.format(bn6.get_shape().as_list()))

        # tf.add_to_collection('gen_params_conv', conv6.w_fc)
        # tf.add_to_collection('gen_params_conv', conv6.w_conv)
        # tf.add_to_collection('gen_params_conv', conv6.b_conv)
        # tf.add_to_collection('gen_params_bn', bn6_layer.scale)
        # tf.add_to_collection('gen_params_bn', bn6_layer.beta)
        # tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv6.w_conv))
        # tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv6.w_fc))

        # decoder
        deconv4 = DeconvLayer(inputs=bn6,
                              filter_shape=[3, 3, 512, 4000],
                              output_shape=conv5.output.get_shape().as_list(),
                              padding='SAME',
                              stride=2,
                              name='deconv4')
        debn4_layer = BatchNormLayer(deconv4.output, is_training, name='debn4')
        debn4 = tf.nn.relu(debn4_layer.output)

        tf.add_to_collection('gen_params_conv', deconv4.w)
        tf.add_to_collection('gen_params_conv', deconv4.b)
        tf.add_to_collection('gen_params_bn', debn4_layer.scale)
        tf.add_to_collection('gen_params_bn', debn4_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(deconv4.w))
        print('deconv4 shape:{}'.format(debn4.get_shape().as_list()))

        deconv3 = DeconvLayer(inputs=debn4,
                              filter_shape=[3, 3, 256, 512],
                              output_shape=conv4.output.get_shape().as_list(),
                              padding='SAME',
                              stride=2,
                              name='deconv3')
        debn3_layer = BatchNormLayer(deconv3.output, is_training, name='debn3')
        debn3 = tf.nn.relu(debn3_layer.output)

        tf.add_to_collection('gen_params_conv', deconv3.w)
        tf.add_to_collection('gen_params_conv', deconv3.b)
        tf.add_to_collection('gen_params_bn', debn3_layer.scale)
        tf.add_to_collection('gen_params_bn', debn3_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(deconv3.w))
        print('deconv3 shape:{}'.format(debn3.get_shape().as_list()))

        deconv2 = DeconvLayer(inputs=debn3,
                              filter_shape=[3, 3, 128, 256],
                              output_shape=conv3.output.get_shape().as_list(),
                              padding='SAME',
                              stride=2,
                              name='deconv2')
        debn2_layer = BatchNormLayer(deconv2.output, is_training, name='debn2')
        debn2 = tf.nn.relu(debn2_layer.output)

        tf.add_to_collection('gen_params_conv', deconv2.w)
        tf.add_to_collection('gen_params_conv', deconv2.b)
        tf.add_to_collection('gen_params_bn', debn2_layer.scale)
        tf.add_to_collection('gen_params_bn', debn2_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(deconv2.w))
        print('deconv2 shape:{}'.format(debn2.get_shape().as_list()))

        deconv1 = DeconvLayer(inputs=debn2,
                              filter_shape=[3, 3, 64, 128],
                              output_shape=conv2.output.get_shape().as_list(),
                              padding='SAME',
                              stride=2,
                              name='deconv1')
        debn1_layer = BatchNormLayer(deconv1.output, is_training, name='debn1')
        debn1 = tf.nn.relu(debn1_layer.output)

        tf.add_to_collection('gen_params_conv', deconv1.w)
        tf.add_to_collection('gen_params_conv', deconv1.b)
        tf.add_to_collection('gen_params_bn', debn1_layer.scale)
        tf.add_to_collection('gen_params_bn', debn1_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(deconv1.w))
        print('deconv1 shape:{}'.format(debn1.get_shape().as_list()))

        recon = DeconvLayer(inputs=debn1,
                            filter_shape=[3, 3, 3, 64],
                            output_shape=[batch_size, 64, 64, 3],
                            stride=2,
                            name='recon')

        tf.add_to_collection('gen_params_conv', recon.w)
        tf.add_to_collection('gen_params_conv', recon.b)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(recon.w))
        print('recon shape:{}'.format(recon.output.get_shape().as_list()))

    return tf.nn.tanh(recon.output)


def discriminator_without_bn(images, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [3, 3, 3, 64], stride=2, name='conv1')
        # bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        # bn1 = tf.contrib.keras.layers.LeakyReLU()(bn1_layer.output)
        bn1 = tf.contrib.keras.layers.LeakyReLU()(conv1.output)

        tf.add_to_collection('dis_params_conv', conv1.w)
        tf.add_to_collection('dis_params_conv', conv1.b)
        # tf.add_to_collection('dis_params_bn', bn1_layer.scale)
        # tf.add_to_collection('dis_params_bn', bn1_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv1.w))
        print(bn1.get_shape().as_list())

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 128], stride=2, name='conv2')
        # bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        # bn2 = tf.contrib.keras.layers.LeakyReLU()(bn2_layer.output)
        bn2 = tf.contrib.keras.layers.LeakyReLU()(conv2.output)

        tf.add_to_collection('dis_params_conv', conv2.w)
        tf.add_to_collection('dis_params_conv', conv2.b)
        # tf.add_to_collection('dis_params_bn', bn2_layer.scale)
        # tf.add_to_collection('dis_params_bn', bn2_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv2.w))
        print(bn2.get_shape().as_list())

        conv3 = Conv2dLayer(bn2, [3, 3, 128, 256], stride=2, name='conv3')
        # bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        # bn3 = tf.contrib.keras.layers.LeakyReLU()(bn3_layer.output)
        bn3 = tf.contrib.keras.layers.LeakyReLU()(conv3.output)

        tf.add_to_collection('dis_params_conv', conv3.w)
        tf.add_to_collection('dis_params_conv', conv3.b)
        # tf.add_to_collection('dis_params_bn', bn3_layer.scale)
        # tf.add_to_collection('dis_params_bn', bn3_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv3.w))
        print(bn3.get_shape().as_list())

        conv4 = Conv2dLayer(bn3, [3, 3, 256, 512], stride=2, name='conv4')
        # bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        # bn4 = tf.contrib.keras.layers.LeakyReLU()(bn4_layer.output)
        bn4 = tf.contrib.keras.layers.LeakyReLU()(conv4.output)

        tf.add_to_collection('dis_params_conv', conv4.w)
        tf.add_to_collection('dis_params_conv', conv4.b)
        # tf.add_to_collection('dis_params_bn', bn4_layer.scale)
        # tf.add_to_collection('dis_params_bn', bn4_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv4.w))
        print(bn4.get_shape().as_list())

        fc = FCLayer(bn4, output_size=1, name='output')

        tf.add_to_collection('dis_params_conv', fc.w)
        tf.add_to_collection('dis_params_conv', fc.b)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(fc.w))

        output = fc.output

    return output[:, 0]


def discriminator_with_bn(images, is_training, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [3, 3, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.contrib.keras.layers.LeakyReLU()(bn1_layer.output)
        # bn1 = tf.contrib.keras.layers.LeakyReLU()(conv1.output)

        tf.add_to_collection('dis_params_conv', conv1.w)
        tf.add_to_collection('dis_params_conv', conv1.b)
        tf.add_to_collection('dis_params_bn', bn1_layer.scale)
        tf.add_to_collection('dis_params_bn', bn1_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv1.w))
        print(bn1.get_shape().as_list())

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.contrib.keras.layers.LeakyReLU()(bn2_layer.output)
        # bn2 = tf.contrib.keras.layers.LeakyReLU()(conv2.output)

        tf.add_to_collection('dis_params_conv', conv2.w)
        tf.add_to_collection('dis_params_conv', conv2.b)
        tf.add_to_collection('dis_params_bn', bn2_layer.scale)
        tf.add_to_collection('dis_params_bn', bn2_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv2.w))
        print(bn2.get_shape().as_list())

        conv3 = Conv2dLayer(bn2, [3, 3, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.contrib.keras.layers.LeakyReLU()(bn3_layer.output)
        # bn3 = tf.contrib.keras.layers.LeakyReLU()(conv3.output)

        tf.add_to_collection('dis_params_conv', conv3.w)
        tf.add_to_collection('dis_params_conv', conv3.b)
        tf.add_to_collection('dis_params_bn', bn3_layer.scale)
        tf.add_to_collection('dis_params_bn', bn3_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv3.w))
        print(bn3.get_shape().as_list())

        conv4 = Conv2dLayer(bn3, [3, 3, 256, 512], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.contrib.keras.layers.LeakyReLU()(bn4_layer.output)
        # bn4 = tf.contrib.keras.layers.LeakyReLU()(conv4.output)

        tf.add_to_collection('dis_params_conv', conv4.w)
        tf.add_to_collection('dis_params_conv', conv4.b)
        tf.add_to_collection('dis_params_bn', bn4_layer.scale)
        tf.add_to_collection('dis_params_bn', bn4_layer.beta)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(conv4.w))
        print(bn4.get_shape().as_list())

        fc = FCLayer(bn4, output_size=1, name='output')

        tf.add_to_collection('dis_params_conv', fc.w)
        tf.add_to_collection('dis_params_conv', fc.b)
        tf.add_to_collection('weight_decay_dis', tf.nn.l2_loss(fc.w))

        output = fc.output

    return output[:, 0]


if __name__ == '__main__':
    batch_size = 128
    # x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='x')
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='x')
    train_flag = tf.placeholder(tf.bool)
    y = reconstruction(x, train_flag)
    print(y.get_shape().as_list())

    init = tf.global_variables_initializer()

    # a = np.random.rand(batch_size, 128, 128, 3)
    a = np.random.rand(batch_size, 227, 227, 3)
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([tf.reduce_mean(y)],
                       feed_dict={train_flag: True,
                                  x: a}))

    # z = discriminator(y, train_flag)
    # print(z.get_shape().as_list())

    # print(tf.get_collection('weight_decay_gen'))
