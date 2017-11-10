from __future__ import print_function

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
                                     initializer=tf.keras.initializers.glorot_normal())

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
                                     initializer=tf.keras.initializers.glorot_normal())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-2],
                                     initializer=tf.constant_initializer(0.))

            deconv = tf.nn.conv2d_transpose(value=self.inputs,
                                            filter=self.w,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding=padding)

            self.output = activation(tf.nn.bias_add(deconv, self.b))


class DilatedConv2dLayer(object):
    """Construct dilated conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 rate,
                 activation=tf.identity,
                 padding='SAME',
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.keras.initializers.glorot_normal())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.atrous_conv2d(value=self.inputs,
                                                filters=self.w,
                                                rate=rate,
                                                padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))


class FCLayer(object):
    """Construct FC layer."""

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
                                     initializer=tf.keras.initializers.glorot_normal())
            self.b = tf.get_variable(name='b',
                                     shape=[output_size],
                                     initializer=tf.constant_initializer(0.))

            self.output = activation(tf.nn.bias_add(tf.matmul(x, self.w), self.b))


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


def completion_network(images, is_training):
    """Construct completion network."""
    batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    bn_layers = []

    with tf.variable_scope('generator'):
        # conv_layers = []
        # bn_layers = []

        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=1, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1_layer.output)  # N, 256, 256, 64
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        tf.add_to_collection('gen_params_conv', conv1.w)
        tf.add_to_collection('gen_params_conv', conv1.b)
        tf.add_to_collection('gen_params_bn', bn1_layer.scale)
        tf.add_to_collection('gen_params_bn', bn1_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv1.w))
        print('conv1 shape:{}'.format(bn1.get_shape().as_list()))

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        tf.add_to_collection('gen_params_conv', conv2.w)
        tf.add_to_collection('gen_params_conv', conv2.b)
        tf.add_to_collection('gen_params_bn', bn2_layer.scale)
        tf.add_to_collection('gen_params_bn', bn2_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv2.w))
        print('conv2 shape:{}'.format(bn2.get_shape().as_list()))

        conv3 = Conv2dLayer(bn2, [3, 3, 128, 128], stride=1, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        tf.add_to_collection('gen_params_conv', conv3.w)
        tf.add_to_collection('gen_params_conv', conv3.b)
        tf.add_to_collection('gen_params_bn', bn3_layer.scale)
        tf.add_to_collection('gen_params_bn', bn3_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv3.w))
        print('conv3 shape:{}'.format(bn3.get_shape().as_list()))

        conv4 = Conv2dLayer(bn3, [3, 3, 128, 256], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        tf.add_to_collection('gen_params_conv', conv4.w)
        tf.add_to_collection('gen_params_conv', conv4.b)
        tf.add_to_collection('gen_params_bn', bn4_layer.scale)
        tf.add_to_collection('gen_params_bn', bn4_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv4.w))
        print('conv4 shape:{}'.format(bn4.get_shape().as_list()))

        conv5 = Conv2dLayer(bn4, [3, 3, 256, 256], stride=1, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        tf.add_to_collection('gen_params_conv', conv5.w)
        tf.add_to_collection('gen_params_conv', conv5.b)
        tf.add_to_collection('gen_params_bn', bn5_layer.scale)
        tf.add_to_collection('gen_params_bn', bn5_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv5.w))
        print('conv5 shape:{}'.format(bn5.get_shape().as_list()))

        conv6 = Conv2dLayer(bn5, [3, 3, 256, 256], stride=1, name='conv6')
        bn6_layer = BatchNormLayer(conv5.output, is_training, name='bn6')
        bn6 = tf.nn.relu(bn6_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv6)
        bn_layers.append(bn6_layer)

        tf.add_to_collection('gen_params_conv', conv6.w)
        tf.add_to_collection('gen_params_conv', conv6.b)
        tf.add_to_collection('gen_params_bn', bn6_layer.scale)
        tf.add_to_collection('gen_params_bn', bn6_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv6.w))
        print('conv6 shape:{}'.format(bn6.get_shape().as_list()))

        dilated_conv7 = DilatedConv2dLayer(bn6, [3, 3, 256, 256], rate=2, name='dilated_conv7')
        bn7_layer = BatchNormLayer(dilated_conv7.output, is_training, name='bn7')
        bn7 = tf.nn.relu(bn7_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv7)
        bn_layers.append(bn7_layer)
        tf.add_to_collection('gen_params_conv', dilated_conv7.w)
        tf.add_to_collection('gen_params_conv', dilated_conv7.b)
        tf.add_to_collection('gen_params_bn', bn7_layer.scale)
        tf.add_to_collection('gen_params_bn', bn7_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(dilated_conv7.w))
        print('dilated_conv7 shape:{}'.format(bn7.get_shape().as_list()))

        dilated_conv8 = DilatedConv2dLayer(bn7, [3, 3, 256, 256], rate=4, name='dilated_conv8')
        bn8_layer = BatchNormLayer(dilated_conv8.output, is_training, name='bn8')
        bn8 = tf.nn.relu(bn8_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv8)
        bn_layers.append(bn8_layer)
        tf.add_to_collection('gen_params_conv', dilated_conv8.w)
        tf.add_to_collection('gen_params_conv', dilated_conv8.b)
        tf.add_to_collection('gen_params_bn', bn8_layer.scale)
        tf.add_to_collection('gen_params_bn', bn8_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(dilated_conv8.w))
        print('dilated_conv8 shape:{}'.format(bn8.get_shape().as_list()))

        dilated_conv9 = DilatedConv2dLayer(bn8, [3, 3, 256, 256], rate=8, name='dilated_conv9')
        bn9_layer = BatchNormLayer(dilated_conv9.output, is_training, name='bn9')
        bn9 = tf.nn.relu(bn9_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv9)
        bn_layers.append(bn9_layer)
        tf.add_to_collection('gen_params_conv', dilated_conv9.w)
        tf.add_to_collection('gen_params_conv', dilated_conv9.b)
        tf.add_to_collection('gen_params_bn', bn9_layer.scale)
        tf.add_to_collection('gen_params_bn', bn9_layer.beta)
        tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(dilated_conv9.w))
        print('dilated_conv9 shape:{}'.format(bn9.get_shape().as_list()))

        dilated_conv10 = DilatedConv2dLayer(bn9, [3, 3, 256, 256], rate=16, name='dilated_conv10')
        bn10_layer = BatchNormLayer(dilated_conv10.output, is_training, name='bn10')
        bn10 = tf.nn.relu(bn10_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv10)
        bn_layers.append(bn10_layer)

        conv11 = Conv2dLayer(bn10, [3, 3, 256, 256], stride=1, name='conv11')
        bn11_layer = BatchNormLayer(conv11.output, is_training, name='bn11')
        bn11 = tf.nn.relu(bn11_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv11)
        bn_layers.append(bn11_layer)

        conv12 = Conv2dLayer(bn11, [3, 3, 256, 256], stride=1, name='conv12')
        bn12_layer = BatchNormLayer(conv12.output, is_training, name='bn12')
        bn12 = tf.nn.relu(bn12_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv12)
        bn_layers.append(bn12_layer)

        deconv13 = DeconvLayer(inputs=bn12,
                               filter_shape=[4, 4, 128, 256],
                               output_shape=[batch_size, 128, 128, 128],
                               stride=2,
                               name='deconv13')
        bn13_layer = BatchNormLayer(deconv13.output, is_training, name='bn13')
        bn13 = tf.nn.relu(bn13_layer.output)  # N, 128, 128, 128
        conv_layers.append(deconv13)
        bn_layers.append(bn13_layer)

        conv14 = Conv2dLayer(bn13, [3, 3, 128, 128], stride=1, name='conv14')
        bn14_layer = BatchNormLayer(conv14.output, is_training, name='bn14')
        bn14 = tf.nn.relu(bn14_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv14)
        bn_layers.append(bn14_layer)

        deconv15 = DeconvLayer(inputs=bn14,
                               filter_shape=[4, 4, 64, 128],
                               output_shape=[batch_size, 256, 256, 64],
                               stride=2,
                               name='deconv15')
        bn15_layer = BatchNormLayer(deconv15.output, is_training, name='bn15')
        bn15 = tf.nn.relu(bn15_layer.output)  # N, 256, 256, 64
        conv_layers.append(deconv15)
        bn_layers.append(bn15_layer)

        conv16 = Conv2dLayer(bn15, [3, 3, 64, 32], stride=1, name='conv16')
        bn16_layer = BatchNormLayer(conv16.output, is_training, name='bn16')
        bn16 = tf.nn.relu(bn16_layer.output)  # N, 256, 256, 32
        conv_layers.append(conv16)
        bn_layers.append(bn16_layer)

        conv17 = Conv2dLayer(bn16, [3, 3, 32, 3], stride=1, name='conv17')
        conv_layers.append(conv17)

    return tf.nn.tanh(conv17.output)  # N, 256, 256, 3
