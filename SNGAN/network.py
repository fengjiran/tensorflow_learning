import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class GAN(object):
    """Construct GAN."""

    model_name = 'GAN'  # name for checkpoint

    def __init__(self, sess, args):
        self.sess = sess
        self.dataset = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.sn = args.sn

        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num

        self.img_size = args.img_size

        self.inputs = None
        self.z = None

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        if self.dataset == 'mnist':
            self.c_dim = 1
            self.data_X = load_mnist(size=self.img_size)

        elif self.dataset == 'ciar10':
            self.c_dim = 3
            self.data_X = load_cifar10(size=self.img_size)

        else:
            self.c_dim = 3
            self.data_X = load_data(dataset_name=self.dataset, size=self.img_size)

        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size

    def discriminator(self, x, is_training=True, reuse=None):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 32

            for i in range(5):
                # ch : 64 -> 128 -> 256 -> 512 -> 1024
                # size : 32 -> 16 -> 8 -> 4 -> 2
                x = conv(x, channels=2 * ch, kernel=5, stride=2, pad=2, sn=self.sn, scope='conv_' + str(i + 1))
                x = batch_norm(x, is_training, scope='bn_' + str(i))
                x = tf.nn.leaky_relu(x)

                ch = ch * 2

            # [bs, 4, 4, 1024]
            x = flatten(x)
            x = fully_conneted(x, 1, sn=self.sn)

            return x

    def generator(self, z, is_training=True, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            ch = 1024

            x = fully_conneted(z, ch)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, 1, 1, ch])

            for i in range(5):
                x = deconv(x, channels=ch // 2, kernel=5, stride=2, scope='deconv_' + str(i + 1))
                x = batch_norm(x, is_training, scope='bn_' + str(i))
                x = tf.nn.relu(x)
                ch = ch // 2

            x = deconv(x, channels=self.c_dim, kernel=5, stride=2, scope='generated_image')
            # [bs, 64, 64, c_dim]

            x = tf.nn.tanh(x)
            return x

    def build_model(self):
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size,
                                                  self.img_size, self.c_dim], name='real_image')
        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        # loss functions
