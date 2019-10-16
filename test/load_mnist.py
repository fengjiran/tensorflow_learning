import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_mnist_image(path, filename, types='train'):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0
    magic, num, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        im = im.reshape(28, 28)
        im = Image.fromarray(im)
        if types == 'train':
            isExists = os.path.exists('./train')
            if not isExists:
                os.mkdir('./train')
            im.save('./train/train_%s.bmp' % image, 'bmp')
        if types == 'test':
            isExists = os.path.exists('./test')
            if not isExists:
                os.mkdir('./test')
            im.save('./test/test_%s.bmp' % image, 'bmp')


def load_mnist_label(path, filename, types='train'):
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

    if types == 'train':
        np.savetxt('./train_labels.csv', Labels, fmt='%i', delimiter=',')
    if types == 'test':
        np.savetext('./test_labels.csv', Labels, fmt='%i', delimiter=',')

    return Labels


if __name__ == '__main__':
    path = 'F://Datasets//mnist//'
    train_images = 'train-images.idx3-ubyte'
    load_mnist_image(path, train_images, 'train')
