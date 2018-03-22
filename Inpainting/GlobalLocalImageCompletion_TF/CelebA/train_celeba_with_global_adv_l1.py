from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import Conv2dLayer
from utils import BatchNormLayer
from utils import FCLayer
from models import completion_network

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l1'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_with_global_adv_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/imagenet_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_without_adv_l1'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_with_global_adv_l1'

# isFirstTimeTrain = False
isFirstTimeTrain = True
batch_size = 32
weight_decay_rate = 1e-4
init_lr = 3e-4
lr_decay_steps = 1000
iters_c = 90000


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
        img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)

        # input image range from -1 to 1
        img = 2 * img - 1

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

        fc7 = FCLayer(bn6, 1, name='fc7')
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

    return fc7.output[:, 0]
