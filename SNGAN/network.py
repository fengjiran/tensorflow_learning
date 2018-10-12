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
