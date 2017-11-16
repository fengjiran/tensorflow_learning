from __future__ import division
from __future__ import print_function

import os
import platform
import numpy as np
import pandas as pd
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
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
elif platform.system() == 'Linux':
    train_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    val_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    val_annotation_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val'
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'


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

    new_short_edge = np.random.randint(256, 384 + 1)
    ratio = new_short_edge / short_edge
    rescaled_img = skimage.transform.rescale(img, ratio)
    # new_long_edge = max(rescaled_img.shape[:2])

    random_y = np.random.randint(0, rescaled_img.shape[0] - height + 1)
    random_x = np.random.randint(0, rescaled_img.shape[1] - width + 1)

    patch = rescaled_img[random_y:random_y + height, random_x:random_x + width, :]

    return patch * 2 - 1  # 256*256 range:[-1,1]


def crop_image_with_hole(image):
    """Generate a single image with hole."""
    image_height, image_width = image.shape[:2]

    hole_height = np.random.randint(96, 128)
    hole_width = np.random.randint(96, 128)

    y = np.random.randint(0, image_height - hole_height)
    x = np.random.randint(0, image_width - hole_width)

    image_with_hole = image.copy()

    # hole = image[y:y + hole_height, x:x + hole_width, :]
    # hole = hole.copy()

    image_with_hole[y:y + hole_height, x:x + hole_width, 0] = 2 * 117. / 255. - 1.
    image_with_hole[y:y + hole_height, x:x + hole_width, 1] = 2 * 104. / 255. - 1.
    image_with_hole[y:y + hole_height, x:x + hole_width, 2] = 2 * 123. / 255. - 1.

    mask_c = np.lib.pad(np.ones([hole_height, hole_width]),
                        pad_width=((y, image_height - hole_height - y), (x, image_width - hole_width - x)),
                        mode='constant')
    mask_c = np.reshape(mask_c, [image_height, image_width, 1])
    mask_c = np.concatenate([mask_c] * 3, 2)

    # generate the location of 128*128 patch for local discriminator
    x_loc = x - int((128 - hole_width) / 2) if x > int((128 - hole_width) / 2) else x
    y_loc = y - int((128 - hole_height) / 2) if y > int((128 - hole_height) / 2) else y

    return image_with_hole, mask_c, x_loc, y_loc  # hole_height, hole_width, y, x


def read_batch(paths):
    images_ori = list(map(load_image, paths))
    images_crops = map(crop_image_with_hole, images_ori)
    images_with_hole, masks, x_locs, y_locs = zip(*images_crops)

    images_ori = np.array(images_ori)
    images_with_hole = np.array(images_with_hole)
    masks = np.array(masks)
    x_locs = np.array(x_locs)
    y_locs = np.array(y_locs)

    return images_ori, images_with_hole, masks, x_locs, y_locs


def create_local_dis_mask(batch_size, x_loc, y_loc):
    pass


if __name__ == '__main__':
    # path = 'C:\\Users\\Richard\\Desktop\\ILSVRC2012_test_00000003.JPEG'
    # test = load_image(path)
    # image_with_hole, mask, x_loc, y_loc = crop_image_with_hole(test)

    # crop = test[y_loc:y_loc + 128, x_loc:x_loc + 128, :]

    # test = (255. * (test + 1) / 2.).astype('uint8')
    # image_with_hole = (255. * (image_with_hole + 1) / 2.).astype('uint8')
    # mask = (255. * (mask + 1) / 2.).astype('uint8')
    # crop = (255. * (crop + 1) / 2.).astype('uint8')

    # print(image_with_hole.shape)
    # print(mask.shape)
    # print(crop.shape)

    # plt.subplot(141)
    # plt.imshow(test)

    # plt.subplot(142)
    # plt.imshow(image_with_hole)

    # plt.subplot(143)
    # plt.imshow(mask)

    # plt.subplot(144)
    # plt.imshow(crop)

    # plt.show()

    images = tf.placeholder(tf.float32, [5, 256, 256, 3])
    x_locs = tf.placeholder(tf.int32, [5], name='x')
    y_locs = tf.placeholder(tf.int32, [5], name='y')

    crops = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0], args[1], args[2], 128, 128),
                      elems=(images, y_locs, x_locs),
                      dtype=tf.float32)

    train_path = pd.read_pickle(compress_path)
    train_path.index = range(len(train_path))
    train_path = train_path.ix[np.random.permutation(len(train_path))]
    # train_path = train_path.ix[range(len(train_path))]

    image_paths = train_path[0:5]['image_path'].values
    a, b, masks, x_locs_, y_locs_ = read_batch(image_paths)

    with tf.Session() as sess:
        results = sess.run(crops, feed_dict={images: a,
                                             x_locs: x_locs_,
                                             y_locs: y_locs_})

        plt.subplot(121)
        plt.imshow((255. * (a[0] + 1) / 2.).astype('uint8'))

        plt.subplot(122)
        plt.imshow((255. * (results[0] + 1) / 2.).astype('uint8'))

        plt.show()

    # c = (255. * (a[0] + 1) / 2.).astype('uint8')
    # d = (255. * (b[0] + 1) / 2.).astype('uint8')

    # plt.subplot(121)
    # plt.imshow(c)

    # plt.subplot(122)
    # plt.imshow(d)

    # plt.show()
