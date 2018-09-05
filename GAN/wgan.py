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
        with tf.variable_scope('discriminator', reuse=reuse):
            pass
