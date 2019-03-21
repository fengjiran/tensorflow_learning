import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import tensorflow as tf

path = 'F:\\Datasets\\qd_imd\\train\\00002_train.png'
img = imread(path)
img1 = imresize(img, [256, 256])
img1 = (img1 > 3).astype(np.uint8) * 255


a = tf.read_file(path)
b = tf.image.decode_png(a)
print(b.get_shape().as_list())
b = tf.reshape(b, [1, 512, 512, 1])
b = tf.image.resize_area(b, [256, 256])
b = tf.cast(tf.greater(b, 3), dtype=tf.uint8) * 255  # {0, 255}
print(b.get_shape().as_list())

with tf.Session() as sess:
    re = sess.run(b)
    print(re)
    re = np.reshape(re, [256, 256])

    plt.figure(figsize=(8, 3))

    plt.subplot(131)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('origin', fontsize=20)

    plt.subplot(132)
    plt.imshow(img1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('resize', fontsize=20)

    plt.subplot(133)
    plt.imshow(re, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('resize2', fontsize=20)

    plt.show()
