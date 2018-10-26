from __future__ import division
import tensorflow as tf
from ops import batch_norm
from ops import instance_norm


class CycleGAN(object):
    """Build cyclegan model."""

    def __init__(self):
        self.activation = tf.nn.leaky_relu
        self.conv_init = tf.truncated_normal_initializer(stddev=0.02)

    def discriminator(self, image, reuse=None, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            h0 = self.activation(tf.layers.conv2d(image, 64, 4,
                                                  strides=2,
                                                  padding='same',
                                                  kernel_initializer=self.conv_init,
                                                  name='d_h0_conv'))
            h1 = self.activation(instance_norm(tf.layers.conv2d(h0, 64 * 2, 4,
                                                                strides=2,
                                                                padding='same',
                                                                kernel_initializer=self.conv_init,
                                                                name='d_h1_conv'), 'd_bn1'))
            h2 = self.activation(instance_norm(tf.layers.conv2d(h1, 64 * 4, 4,
                                                                strides=2,
                                                                padding='same',
                                                                kernel_initializer=self.conv_init,
                                                                name='d_h2_conv'), 'd_bn2'))
            h3 = self.activation(instance_norm(tf.layers.conv2d(h2, 64 * 8, 4,
                                                                strides=2,
                                                                padding='same',
                                                                kernel_initializer=self.conv_init,
                                                                name='d_h3_conv'), 'd_bn3'))
            h4 = tf.layers.conv2d(h3, 1, 4,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=self.conv_init,
                                  name='d_h4_pred')

            return h4

    def generator_unet(self, image, reuse=None, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            pass
