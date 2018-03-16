from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import BatchNormLayer
from utils import FCLayer


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

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [3, 3, 128, 128], stride=1, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [3, 3, 128, 256], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [3, 3, 256, 256], stride=1, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        conv6 = Conv2dLayer(bn5, [3, 3, 256, 256], stride=1, name='conv6')
        bn6_layer = BatchNormLayer(conv5.output, is_training, name='bn6')
        bn6 = tf.nn.relu(bn6_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv6)
        bn_layers.append(bn6_layer)

        # Dilated conv from here
        dilated_conv7 = DilatedConv2dLayer(bn6, [3, 3, 256, 256], rate=2, name='dilated_conv7')
        # bn7_layer = BatchNormLayer(dilated_conv7.output, is_training, name='bn7')
        # bn7 = tf.nn.relu(bn7_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv7)
        # bn_layers.append(bn7_layer)

        dilated_conv8 = DilatedConv2dLayer(dilated_conv7.output, [3, 3, 256, 256], rate=4, name='dilated_conv8')
        # bn8_layer = BatchNormLayer(dilated_conv8.output, is_training, name='bn8')
        # bn8 = tf.nn.relu(bn8_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv8)
        # bn_layers.append(bn8_layer)

        dilated_conv9 = DilatedConv2dLayer(dilated_conv8.output, [3, 3, 256, 256], rate=8, name='dilated_conv9')
        # bn9_layer = BatchNormLayer(dilated_conv9.output, is_training, name='bn9')
        # bn9 = tf.nn.relu(bn9_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv9)
        # bn_layers.append(bn9_layer)

        dilated_conv10 = DilatedConv2dLayer(dilated_conv9.output, [3, 3, 256, 256], rate=16, name='dilated_conv10')
        # bn10_layer = BatchNormLayer(dilated_conv10.output, is_training, name='bn10')
        # bn10 = tf.nn.relu(bn10_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv10)
        # bn_layers.append(bn10_layer)

        # resize back
        conv11 = Conv2dLayer(dilated_conv10.output, [3, 3, 256, 256], stride=1, name='conv11')
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
                               output_shape=[batch_size, conv2.output_shape[1],
                                             conv2.output_shape[2], 128],
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
                               output_shape=[batch_size, conv1.output_shape[1],
                                             conv1.output_shape[2], 64],
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

        print('Print the completion network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('gen_params_conv', conv_layer.w)
            tf.add_to_collection('gen_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('gen_params_bn', bn_layer.scale)
            tf.add_to_collection('gen_params_bn', bn_layer.beta)

    return tf.nn.tanh(conv17.output)  # N, 256, 256, 3


def global_discriminator(images, is_training, reuse=None):
    """Construct global discriminator network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    bn_layers = []
    with tf.variable_scope('global_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1_layer.output)
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [5, 5, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [5, 5, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [5, 5, 256, 512], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [5, 5, 512, 512], stride=2, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        conv6 = Conv2dLayer(bn5, [5, 5, 512, 512], stride=2, name='conv6')
        bn6_layer = BatchNormLayer(conv6.output, is_training, name='bn6')
        bn6 = tf.nn.relu(bn6_layer.output)
        conv_layers.append(conv6)
        bn_layers.append(bn6_layer)

        fc7 = FCLayer(bn6, 1024, activation=tf.nn.relu, name='fc7')
        conv_layers.append(fc7)

        print('Print the global discriminator network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('global_dis_params_conv', conv_layer.w)
            tf.add_to_collection('global_dis_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_global_dis', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('global_dis_params_bn', bn_layer.scale)
            tf.add_to_collection('global_dis_params_bn', bn_layer.beta)

    return fc7.output


def local_discriminator(images, is_training, reuse=None):
    """Construct local discriminator network."""
    conv_layers = []
    bn_layers = []
    with tf.variable_scope('local_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1_layer.output)
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [5, 5, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [5, 5, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [5, 5, 256, 512], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [5, 5, 512, 512], stride=2, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        fc6 = FCLayer(bn5, 1024, activation=tf.nn.relu, name='fc6')
        conv_layers.append(fc6)

        print('Print the local discriminator network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('local_dis_params_conv', conv_layer.w)
            tf.add_to_collection('local_dis_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_local_dis', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('local_dis_params_bn', bn_layer.scale)
            tf.add_to_collection('local_dis_params_bn', bn_layer.beta)

    return fc6.output


def combine_discriminator(global_inputs, local_inputs, is_training, reuse=None):
    """Combine the global and local discriminators."""
    global_dis = global_discriminator(global_inputs, is_training, reuse=reuse)
    local_dis = local_discriminator(local_inputs, is_training, reuse=reuse)

    x = tf.concat([global_dis, local_dis], axis=1)
    with tf.variable_scope('combine_discriminator', reuse=reuse):
        fc = FCLayer(x, 1, name='output')
        tf.add_to_collection('combine_dis_params', fc.w)
        tf.add_to_collection('combine_dis_params', fc.b)
        tf.add_to_collection('weight_decay_combine_dis', tf.nn.l2_loss(fc.w))

    return fc.output[:, 0]


if __name__ == '__main__':
    batch_size = 128
    x = tf.placeholder(tf.float32, [batch_size, 128, 255, 3], name='x')
    global_inputs = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], name='global_inputs')
    local_inputs = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='local_inputs')
    train_flag = tf.placeholder(tf.bool)

    y = completion_network(x, train_flag)
    # y = global_discriminator(x, train_flag)
    # y = local_discriminator(x, train_flag)
    # y = combine_discriminator(global_inputs, local_inputs, train_flag)
    init = tf.global_variables_initializer()

    a = np.random.rand(batch_size, 128, 255, 3)
    b = np.random.rand(batch_size, 256, 256, 3)  # for global dis
    c = np.random.rand(batch_size, 128, 128, 3)  # for local dis

    with tf.Session() as sess:
        sess.run(init)
        # print(sess.run([tf.reduce_mean(y)],
        #                feed_dict={train_flag: True,
        #                           global_inputs: b,
        #                           local_inputs: c}))
        print(sess.run([tf.reduce_mean(y)], feed_dict={train_flag: True, x: a}))
