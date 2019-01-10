from __future__ import print_function
import tensorflow as tf


class InpaintingModel(object):
    """Construct model."""

    def __init__(self):
        print('Construct the inpainting model.')

    def edge_generator(self, x):
        pass

    def inpaint_generator(self, x):
        pass

    def instance_norm(self, x, name="instance_norm"):
        with tf.variable_scope(name):
            depth = x.get_shape()[3]
            scale = tf.get_variable("scale", [depth],
                                    initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth],
                                     initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (x - mean) * inv
            return scale * normalized + offset

    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape().as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable('u', [1, w_shape[-1]],
                            initializer=tf.random_normal_initializer(),
                            trainable=False)

        u_hat = u
        v_hat = None

        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


def residual_block(x, in_channels, out_channels, dilation=1, name='residual_block'):
    pass


def spectral_norm(w, iteration=1):
    w_shape = w.shape().as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)

    u_hat = u
    v_hat = None

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
