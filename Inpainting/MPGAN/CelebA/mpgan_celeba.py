from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import BatchNormLayer
from utils import FCLayer

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l1'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\models_global_local_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_without_adv_l1'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/models_global_local_l1'

# isFirstTimeTrain = False
isFirstTimeTrain = True
batch_size = 32
weight_decay_rate = 1e-4
init_lr_g = 3e-4
init_lr_d = 3e-4
lr_decay_steps = 1000
iters_total = 100000
iters_d = 10000
alpha_rec = 0.995
alpha_glo = 0.0025
alpha_mp = 0.0025

gt_height = 96
gt_width = 96


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 48
        high = 96
        image_height = 178
        image_width = 178
        gt_height = 96
        gt_width = 96

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        img = tf.cast(img_decoded, tf.float32)
        # img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)

        # input image range from -1 to 1
        # img = 2 * img - 1
        img = img / 127.5 - 1

        ori_image = tf.identity(img)

        hole_height, hole_width = np.random.randint(low, high, size=(2))
        y = tf.random_uniform([], 0, image_height - hole_height, tf.int32)
        x = tf.random_uniform([], 0, image_width - hole_width, tf.int32)

        mask = tf.pad(tensor=tf.ones((hole_height, hole_width)),
                      paddings=[[y, image_height - hole_height - y],
                                [x, image_width - hole_width - x]])
        mask = tf.reshape(mask, [image_height, image_width, 1])
        mask = tf.concat([mask] * 3, 2)

        image_with_hole = img * (1 - mask) + mask

        # generate the location of 96*96 patch for local discriminator
        x_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, x + hole_width - gt_width]),
                                  maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
                                  dtype=tf.int32)
        y_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, y + hole_height - gt_height]),
                                  maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
                                  dtype=tf.int32)

        return ori_image, image_with_hole, mask, x_loc, y_loc


def completion_network(images, is_training, batch_size):
    """Construct completion network."""
    # batch_size = images.get_shape().as_list()[0]
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
        bn1 = tf.nn.leaky_relu(bn1_layer.output)
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [5, 5, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.leaky_relu(bn2_layer.output)
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [5, 5, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.leaky_relu(bn3_layer.output)
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [5, 5, 256, 512], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.leaky_relu(bn4_layer.output)
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [5, 5, 512, 512], stride=2, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.leaky_relu(bn5_layer.output)
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        conv6 = Conv2dLayer(bn5, [5, 5, 512, 512], stride=2, name='conv6')
        bn6_layer = BatchNormLayer(conv6.output, is_training, name='bn6')
        bn6 = tf.nn.leaky_relu(bn6_layer.output)
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


def markovian_discriminator(images, is_training, reuse=None):
    """Construct markovian discriminator."""
    conv_layers = []
    bn_layers = []
    with tf.variable_scope('markovian_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.leaky_relu(bn1_layer.output, alpha=0.2)
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [5, 5, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.leaky_relu(bn2_layer.output, alpha=0.2)
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [5, 5, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.leaky_relu(bn3_layer.output, alpha=0.2)
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [5, 5, 256, 1], stride=1, name='conv4')
        conv_layers.append(conv4)

        print('Print the local discriminator network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('local_dis_params_conv', conv_layer.w)
            tf.add_to_collection('local_dis_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_local_dis', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('local_dis_params_bn', bn_layer.scale)
            tf.add_to_collection('local_dis_params_bn', bn_layer.beta)

    return conv4.output


if __name__ == '__main__':
    batch_size = 100
    imgs = tf.placeholder(tf.float32, [100, 96, 96, 3])
    train_flag = tf.placeholder(tf.bool)

    result = markovian_discriminator(imgs, train_flag)
    print(result.get_shape())
