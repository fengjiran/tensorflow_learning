from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf

from utils import load_mnist


class WGAN(object):
    """Construct WGAN class."""

    model_name = 'WGAN'  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim
            self.c_dim = 1

            self.disc_iters = 1  # The number of critic iterations for one step generator

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size

        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=None):
        axis = list(range(len(x.get_shape()) - 1))
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.conv2d(x, 64, 4, 2, padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name='d_conv1')
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 128, 4, 2, padding='same',
                                 kernel_initializer=tf.keras.initializers.glorot_normal(), name='d_conv2')
            x = tf.layers.batch_normalization(x, axis=axis, training=is_training, name='d_bn2')
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, [self.batch_size, -1])
            x = tf.layers.dense(x, 1024, kernel_initializer=tf.keras.initializers.glorot_normal(), name='d_fc3')
            x = tf.layers.batch_normalization(x, axis=axis, training=is_training, name='d_bn3')
            x = tf.nn.leaky_relu(x)
            out_logit = tf.layers.dense(x, 1, kernel_initializer=tf.keras.initializers.glorot_normal(), name='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, x

    def generator(self, z, is_training=True, reuse=None):
        axis = list(range(len(z.get_shape()) - 1))
        with tf.variable_scope('generator', reuse=reuse):
            x = tf.layers.dense(z, 1024, kernel_initializer=tf.keras.initializers.glorot_normal(), name='g_fc1')
            x = tf.layers.batch_normalization(x, axis=axis, training=is_training, name='g_bn1')
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 128 * 7 * 7, kernel_initializer=tf.keras.initializers.glorot_normal(), name='g_fc2')
            x = tf.layers.batch_normalization(x, axis=axis, training=is_training, name='g_bn2')
            x = tf.nn.relu(x)

            x = tf.reshape(x, [self.batch_size, 7, 7, 128])
            x = tf.layers.conv2d_transpose(x, 64, (4, 4), strides=(2, 2), name='g_dc3')
            x = tf.layers.batch_normalization(x, axis=list(range(len(z.get_shape()) - 1)),
                                              training=is_training, name='g_bn3')
            x = tf.nn.relu(x)
