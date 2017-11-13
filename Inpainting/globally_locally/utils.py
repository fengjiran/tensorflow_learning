from __future__ import division
import os
import platform
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
from PIL import Image
import matplotlib.pyplot as plt

if platform.system() == 'Windows':
    train_path = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\train'
    val_path = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\val'
    test_path = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\test'
    val_annotation_path = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Annotations\\CLS-LOC\\val'
elif platform.system() == 'Linux':
    train_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    val_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    val_annotation_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val'


class Conv2dLayer(object):
    """Construct conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.keras.initializers.glorot_normal())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.conv2d(self.inputs, self.w, [1, stride, stride, 1], padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DeconvLayer(object):
    """Construct deconv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 output_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.keras.initializers.glorot_normal())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-2],
                                     initializer=tf.constant_initializer(0.))

            deconv = tf.nn.conv2d_transpose(value=self.inputs,
                                            filter=self.w,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding=padding)

            self.output = activation(tf.nn.bias_add(deconv, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DilatedConv2dLayer(object):
    """Construct dilated conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 rate,
                 activation=tf.identity,
                 padding='SAME',
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.keras.initializers.glorot_normal())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.atrous_conv2d(value=self.inputs,
                                                filters=self.w,
                                                rate=rate,
                                                padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class FCLayer(object):
    """Construct FC layer."""

    def __init__(self,
                 inputs,
                 output_size,
                 activation=tf.identity,
                 name=None):
        self.inputs = inputs
        shape = inputs.get_shape().as_list()
        input_size = np.prod(shape[1:])
        x = tf.reshape(self.inputs, [-1, input_size])

        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=[input_size, output_size],
                                     initializer=tf.keras.initializers.glorot_normal())
            self.b = tf.get_variable(name='b',
                                     shape=[output_size],
                                     initializer=tf.constant_initializer(0.))

            self.output = activation(tf.nn.bias_add(tf.matmul(x, self.w), self.b))
            self.output_shape = self.output.get_shape().as_list()


class BatchNormLayer(object):
    """Construct batch norm layer."""

    def __init__(self, inputs, is_training, decay=0.999, epsilon=1e-5, name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.scale = tf.get_variable(name='scale',
                                         shape=[inputs.get_shape()[-1]],
                                         initializer=tf.constant_initializer(1.))

            self.beta = tf.get_variable(name='beta',
                                        shape=[inputs.get_shape()[-1]],
                                        initializer=tf.constant_initializer(0.))

            self.pop_mean = tf.get_variable(name='pop_mean',
                                            shape=[inputs.get_shape()[-1]],
                                            initializer=tf.constant_initializer(0.),
                                            trainable=False)

            self.pop_var = tf.get_variable(name='pop_var',
                                           shape=[inputs.get_shape()[-1]],
                                           initializer=tf.constant_initializer(1.),
                                           trainable=False)

            def mean_var_update():
                axes = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs, axes)
                train_mean = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_var_update, lambda: (self.pop_mean, self.pop_var))

            self.output = tf.nn.batch_normalization(x=inputs,
                                                    mean=mean,
                                                    variance=variance,
                                                    offset=self.beta,
                                                    scale=self.scale,
                                                    variance_epsilon=epsilon)


def array_to_image(array):
    r = Image.fromarray(array[0]).convert('L')
    g = Image.fromarray(array[1]).convert('L')
    b = Image.fromarray(array[2]).convert('L')

    image = Image.merge('RGB', (r, g, b))

    return image


def load_image(path, height=256, width=256):
    try:
        img = skimage.io.imread(path).astype(float)
    except TypeError:
        return None

    if img is None:
        return None

    if len(img.shape) < 2:
        return None

    if len(img.shape) == 4:
        return None

    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)

    if img.shape[2] == 4:
        img = img[:, :, 3]

    if img.shape[2] > 4:
        return None

    img /= 255.

    short_edge = min(img.shape[:2])
    # long_edge = max(img.shape[:2])

    new_short_edge = np.random.randint(256, 384)
    ratio = new_short_edge / short_edge
    rescaled_img = skimage.transform.rescale(img, ratio)
    # new_long_edge = max(rescaled_img.shape[:2])

    random_y = np.random.randint(0, rescaled_img.shape[0] - height)
    random_x = np.random.randint(0, rescaled_img.shape[1] - width)

    patch = rescaled_img[random_y:random_y + height, random_x:random_x + width, :]

    return patch * 2 - 1  # 256*256 range:[-1,1]


def crop_image_with_hole(image):
    image_height, image_width = image.shape[:2]
    hole_height = np.random.randint(96, 128)
    hole_width = np.random.randint(96, 128)

    y = np.random.randint(0, image_height - hole_height)
    x = np.random.randint(0, image_width - hole_width)

    image_with_hole = image.copy()

    hole = image[y:y + hole_height, x:x + hole_width, :]
    # hole = hole.copy()

    image_with_hole[y:y + hole_height, x:x + hole_width, 0] = 2 * 117. / 255. - 1.
    image_with_hole[y:y + hole_height, x:x + hole_width, 1] = 2 * 104. / 255. - 1.
    image_with_hole[y:y + hole_height, x:x + hole_width, 2] = 2 * 123. / 255. - 1.

    mask = np.lib.pad(np.ones([hole_height, hole_width]),
                      pad_width=((image_height - hole_height - y, y), (x, image_width - hole_width - x)),
                      mode='constant')
    mask = np.reshape(mask, [image_height, image_width, 1])
    mask = np.concatenate([mask] * 3, 2)

    return image_with_hole, hole, mask  # hole_height, hole_width, y, x


if __name__ == '__main__':
    path = 'C:\\Users\\Richard\\Desktop\\ILSVRC2012_test_00000003.JPEG'
    test = load_image(path)
    # print(test.shape)
    image, hole, mask = crop_image_with_hole(test)
    test = (255. * (test + 1) / 2.).astype('uint8')
    image = (255. * (image + 1) / 2.).astype('uint8')
    hole = (255. * (hole + 1) / 2.).astype('uint8')

    print(image.shape)
    print(hole.shape)

    plt.subplot(131)
    plt.imshow(test)

    plt.subplot(132)
    plt.imshow(image)

    plt.subplot(133)
    plt.imshow(hole)

    plt.show()
