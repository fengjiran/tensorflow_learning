from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm


class InpaintingModel(object):
    """Construct model."""

    def __init__(self):
        print('Construct the inpainting model.')

    def edge_generator(self, x):
        with tf.variable_scope('edge_generator'):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', name='conv3')
            x = instance_norm(x, name='in3')
            x = tf.nn.relu(x)

            # resnet block
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block1')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block2')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block3')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block4')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block5')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block6')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block7')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block8')

            # decoder
            x = deconv(x, channels=128, kernel=4, stride=2, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            x = deconv(x, channels=64, kernel=4, stride=2, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=1, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv4')
            x = tf.nn.sigmoid(x)

            return x

    def edge_discriminator(self, x, reuse=None):
        with tf.variable_scope('edge_discriminator', reuse=reuse):
            x = conv(x, channels=64, kernel=4, stride=2, pad=1, pad_type='zero', name='conv1')
            x = tf.nn.leaky_relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', name='conv2')
            x = tf.nn.leaky_relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', name='conv3')
            x = tf.nn.leaky_relu(x)

    def inpaint_generator(self, x):
        pass


if __name__ == '__main__':
    model = InpaintingModel()

    bs = 10
    x = tf.random_uniform([bs, 256, 256, 3])

    out = model.edge_generator(x)
    print(out.get_shape())
