import os
from glob import glob
import scipy.misc
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist


def load_mnist(size=64, path=None):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data(path)
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # x = np.expand_dims(x, axis=-1)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
    x = np.expand_dims(x, axis=-1)
    return x


def load_cifar10(size=64):
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])

    return x


def load_data(dataset_name, size=64):
    if dataset_name == 'mnist':
        x = load_mnist()
    elif dataset_name == 'cifar10':
        x = load_cifar10()
    else:
        image_list = glob(os.path.join("./data", dataset_name, '*.*'))
        x = np.asarray([preprocessing(image, size) for image in image_list])

    return x


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x


def normalize(x):
    return x / 127.5 - 1
