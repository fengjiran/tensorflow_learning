import os
import struct
import numpy as np


def load_mnist_image(path, filename):
    # load image
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0
    magic, num, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    images = []

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        images.append(im)

    return np.array(images)


def load_mnist_label(path, filename):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0
    magic, num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    Labels = np.zeros(num)

    for i in range(num):
        Labels[i] = np.array(struct.unpack_from('>B', buf, index))
        index += struct.calcsize('>B')

    return Labels.astype(np.int32)


if __name__ == '__main__':
    path = '/Users/richard/Desktop/mnist'
    filename_images = 'train-images-idx3-ubyte'
    filename_labels = 'train-labels-idx1-ubyte'
    train_images = load_mnist_image(path, filename_images)
    train_labels = load_mnist_label(path, filename_labels)
    print(train_images.shape)
    print(train_labels[0])
