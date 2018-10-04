from __future__ import division
from __future__ import print_function

import tensorflow as tf

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d


class ResNet_v1_50(object):
    """Construct resnet v1 50."""

    def __init__(self, inputs, is_training, num_classes=1000, scope='resnet_v1_50'):
        self.inputs = inputs
        self.is_training = is_training
        self.num_classes = num_classes

        with tf.variable_scope(scope):
            x = tf.layers.conv2d(inputs, 64, 7, 2, padding='same',
                                 kernel_initializer=conv2d_initializer(),
                                 name='conv1')
            x = tf.layers.batch_normalization(x,
                                              axis=list(range(len(x.get_shape()) - 1)),
                                              name='bn1')
            x = tf.nn.relu(x)

            x = tf.layers.max_pooling2d(x, 3, strides=2, padding='same', name='maxpool1')

            x = self.block(x, 64, 256, 3, init_stride=1, scope='block2')
            x = self.block(x, 128, 512, 4, scope='block3')
            x = self.block(x, 256, 1024, 6, scope='block4')
            x = self.block(x, 512, 2048, 3, scope='block5')

            x = tf.layers.average_pooling2d(x, 7, 7, name='avgpool')
            x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')
            self.logits = tf.layers.dense(x, self.num_classes, name='fc6')
            self.predictions = tf.nn.softmax(self.logits)

    def bottleneck(self, h, h_out, n_out, stride=None, scope='bottleneck'):
        """Construct a residual bottleneck unit."""
        n_in = h.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            x = tf.layers.conv2d(h, h_out, 1, strides=stride, padding='same',
                                 kernel_initializer=conv2d_initializer(),
                                 name='conv1')
            x = tf.layers.batch_normalization(x,
                                              axis=list(range(len(x.get_shape()) - 1)),
                                              training=self.is_training,
                                              name='bn1')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, h_out, 3, strides=1, padding='same',
                                 kernel_initializer=conv2d_initializer(),
                                 name='conv2')
            x = tf.layers.batch_normalization(x,
                                              axis=list(range(len(x.get_shape()) - 1)),
                                              training=self.is_training,
                                              name='bn2')
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, n_out, 1, strides=1, padding='same',
                                 kernel_initializer=conv2d_initializer(),
                                 name='conv3')
            x = tf.layers.batch_normalization(x,
                                              axis=list(range(len(x.get_shape()) - 1)),
                                              training=self.is_training,
                                              name='bn3')

            if n_in != n_out:
                shortcut = tf.layers.conv2d(h, n_out, 1, strides=stride, padding='same',
                                            kernel_initializer=conv2d_initializer(),
                                            name='conv4')
                shortcut = tf.layers.batch_normalization(shortcut,
                                                         axis=list(range(len(shortcut.get_shape()) - 1)),
                                                         training=self.is_training,
                                                         name='bn4')
            else:
                shortcut = h
            return tf.nn.relu(shortcut + x)

    def block(self, x, n_in, n_out, n_bottleneck, init_stride=2, scope='block'):
        with tf.variable_scope(scope):
            # h_out = n_out//4
            out = self.bottleneck(x, n_in, n_out, stride=init_stride, scope='bottleneck1')

            for i in range(1, n_bottleneck):
                out = self.bottleneck(out, n_in, n_out, scope='bottleneck%s' % (i + 1))

            return out


if __name__ == '__main__':
    x = tf.random_normal([10, 224, 224, 3])
    resnet50 = ResNet_v1_50(x, is_training=True)
    print(resnet50.logits.get_shape())
