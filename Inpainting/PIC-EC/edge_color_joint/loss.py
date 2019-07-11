import os
import inspect
import platform as pf
import numpy as np
import tensorflow as tf

if pf.system() == 'Windows':
    vgg19_npy_path = 'F:\\Datasets\\vgg19.npy'
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        vgg19_npy_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/vgg19.npy'
    elif pf.node() == 'icie-Precision-T7610':
        vgg19_npy_path = '/home/icie/Datasets/vgg19.npy'


def adversarial_loss(inputs, is_real, gan_type='nsgan', is_disc=None):
    """type: nsgan | lsgan | hinge."""
    if gan_type == 'nsgan':
        labels = tf.ones_like(inputs) if is_real else tf.zeros_like(inputs)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels))
    elif gan_type == 'lsgan':
        labels = tf.ones_like(inputs) if is_real else tf.zeros_like(inputs)
        loss = tf.losses.mean_squared_error(predictions=inputs, labels=labels)
    elif gan_type == 'hinge':
        if is_disc:
            if is_real:
                inputs = -inputs
            loss = tf.reduce_mean(tf.nn.relu(1 + inputs))
        else:
            loss = tf.reduce_mean(-inputs)

    return loss


def compute_gram(x):
    shape = x.get_shape()

    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(x, shape=[-1, num_channels])

    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram


# def compute_gram(x):
#     b, ch, h, w = x.size()
#     f = x.view(b, ch, w * h)
#     f_T = f.transpose(1, 2)
#     G = f.bmm(f_T) / (h * w * ch)

#     return G


def style_loss(x, y):
    """Compute style loss, vgg-based."""
    # vgg = Vgg19()
    x_vgg_out = x  # vgg.build(x)
    y_vgg_out = y  # vgg.build(y)

    # compute style loss
    style_loss1 = 0.0
    style_loss1 += tf.losses.absolute_difference(compute_gram(x_vgg_out['relu2_2']),
                                                 compute_gram(y_vgg_out['relu2_2']))

    style_loss1 += tf.losses.absolute_difference(compute_gram(x_vgg_out['relu3_4']),
                                                 compute_gram(y_vgg_out['relu3_4']))

    style_loss1 += tf.losses.absolute_difference(compute_gram(x_vgg_out['relu4_4']),
                                                 compute_gram(y_vgg_out['relu4_4']))

    style_loss1 += tf.losses.absolute_difference(compute_gram(x_vgg_out['relu5_2']),
                                                 compute_gram(y_vgg_out['relu5_2']))

    return style_loss1


def perceptual_loss(x, y, weights=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """Compute perceptual loss, vgg-based."""
    # vgg = Vgg19()
    x_vgg_out = x  # vgg.build(x)
    y_vgg_out = y  # vgg.build(y)

    content_loss = 0.0

    content_loss += weights[0] * tf.losses.absolute_difference(x_vgg_out['relu1_1'], y_vgg_out['relu1_1'])
    content_loss += weights[1] * tf.losses.absolute_difference(x_vgg_out['relu2_1'], y_vgg_out['relu2_1'])
    content_loss += weights[2] * tf.losses.absolute_difference(x_vgg_out['relu3_1'], y_vgg_out['relu3_1'])
    content_loss += weights[3] * tf.losses.absolute_difference(x_vgg_out['relu4_1'], y_vgg_out['relu4_1'])
    content_loss += weights[4] * tf.losses.absolute_difference(x_vgg_out['relu5_1'], y_vgg_out['relu5_1'])

    return content_loss


class Vgg19():
    """Construct VGG19 model."""

    def __init__(self, vgg19_npy_path=vgg19_npy_path):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg19.npy')
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, allow_pickle=True, encoding='latin1').item()
        print('npy file loaded')

    def forward(self, rgb, reuse=None):
        """
        Load variables from npy file to build VGG.

        rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        with tf.variable_scope('vgg', reuse=reuse):
            print('Build model started.')
            rgb = (rgb + 1.0) / 2.0  # [0, 1]
            rgb = tf.image.resize_image_with_crop_or_pad(rgb, 224, 224)
            # rgb = tf.image.central_crop(rgb, 224 / 256)
            VGG_MEAN = [103.939, 116.779, 123.68]
            rgb_scaled = rgb * 255.0

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]

            bgr = tf.concat(axis=3,
                            values=[blue - VGG_MEAN[0],
                                    green - VGG_MEAN[1],
                                    red - VGG_MEAN[2]])

            bgr = bgr / 255.
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

            conv1_1 = self.conv_layer(bgr, 'conv1_1', reuse)
            conv1_2 = self.conv_layer(conv1_1, 'conv1_2', reuse)
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.conv_layer(pool1, 'conv2_1', reuse)
            conv2_2 = self.conv_layer(conv2_1, 'conv2_2', reuse)
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.conv_layer(pool2, "conv3_1", reuse)
            conv3_2 = self.conv_layer(conv3_1, "conv3_2", reuse)
            conv3_3 = self.conv_layer(conv3_2, "conv3_3", reuse)
            conv3_4 = self.conv_layer(conv3_3, "conv3_4", reuse)
            pool3 = self.max_pool(conv3_4, 'pool3')

            conv4_1 = self.conv_layer(pool3, "conv4_1", reuse)
            conv4_2 = self.conv_layer(conv4_1, "conv4_2", reuse)
            conv4_3 = self.conv_layer(conv4_2, "conv4_3", reuse)
            conv4_4 = self.conv_layer(conv4_3, "conv4_4", reuse)
            pool4 = self.max_pool(conv4_4, 'pool4')

            conv5_1 = self.conv_layer(pool4, "conv5_1", reuse)
            conv5_2 = self.conv_layer(conv5_1, "conv5_2", reuse)
            conv5_3 = self.conv_layer(conv5_2, "conv5_3", reuse)
            conv5_4 = self.conv_layer(conv5_3, "conv5_4", reuse)
            # pool5 = self.max_pool(conv5_4, 'pool5')

            # self.fc6 = self.fc_layer(self.pool5, (25088, 4096), "fc6", reuse)
            # assert self.fc6.get_shape().as_list()[1:] == [4096]
            # self.relu6 = tf.nn.relu(self.fc6)

            # self.fc7 = self.fc_layer(self.relu6, (4096, 4096), "fc7", reuse)
            # self.relu7 = tf.nn.relu(self.fc7)

            # self.fc8 = self.fc_layer(self.relu7, (4096, 1000), "fc8", reuse)

            # self.prob = tf.nn.softmax(self.fc8, name="prob")

            print('build model finished!')

            out = {
                'relu1_1': conv1_1,
                'relu1_2': conv1_2,

                'relu2_1': conv2_1,
                'relu2_2': conv2_2,

                'relu3_1': conv3_1,
                'relu3_2': conv3_2,
                'relu3_3': conv3_3,
                'relu3_4': conv3_4,

                'relu4_1': conv4_1,
                'relu4_2': conv4_2,
                'relu4_3': conv4_3,
                'relu4_4': conv4_4,

                'relu5_1': conv5_1,
                'relu5_2': conv5_2,
                'relu5_3': conv5_3,
                'relu5_4': conv5_4
            }

            return out

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            value = self.data_dict[name]

            filt = tf.get_variable(name='kernel',
                                   shape=value[0].shape,
                                   initializer=tf.constant_initializer(value[0]),
                                   trainable=False)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = tf.get_variable(name='bias',
                                          shape=value[1].shape,
                                          initializer=tf.constant_initializer(value[1]),
                                          trainable=False)
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

            return relu

    def fc_layer(self, bottom, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            value = self.data_dict[name]
            shape = bottom.get_shape().as_list()
            dim = 1

            for d in shape[1:]:
                dim *= d

            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable(name='weights',
                                      shape=value[0].shape,
                                      initializer=tf.constant_initializer(value[0]),
                                      trainable=False)
            biases = tf.get_variable(name='bias',
                                     shape=value[1].shape,
                                     initializer=tf.constant_initializer(value[1]),
                                     trainable=False)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc


if __name__ == '__main__':
    vgg = Vgg19()

    a = tf.placeholder(tf.float32, [10, 256, 256, 3])
    out = vgg.forward(a)
    out1 = vgg.forward(a, True)
    vgg.data_dict = None
