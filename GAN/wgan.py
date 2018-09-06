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
            self.inputs = None
            self.z = None
            self.d_loss = None
            self.g_loss = None
            self.d_optim = None
            self.g_optim = None
            self.clip_D = None
            self.fake_images = None
            self.g_sum = None
            self.d_sum = None
            self.sample_z = None
            self.saver = None
            self.writer = None

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
            x = tf.layers.conv2d_transpose(x, 64, (4, 4), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_normal(),
                                           name='g_dc3')
            x = tf.layers.batch_normalization(x, axis=list(range(len(z.get_shape()) - 1)),
                                              training=is_training, name='g_bn3')
            x = tf.nn.relu(x)
            x = tf.layers.conv2d_transpose(x, 1, (4, 4), strides=(2, 2), padding='same',
                                           kernel_initializer=tf.keras.initializers.glorot_normal(),
                                           name='g_dc4')
            out = tf.nn.sigmoid(x)

            return out

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_image')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        # loss function

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True)

        # output of D for fake images
        G = self.generator(self.z, is_training=True)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = -tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = -d_loss_fake

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATA_OPS)):
            self.d_optim = tf.train.AdamOptimizer(
                self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)

            self.g_optim = tf.train.AdamOptimizer(
                self.learning_rate * 5, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        # test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        # summary
        d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
        d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
        d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):
        # initialize all the variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # save the model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore checkpoint if it exits
