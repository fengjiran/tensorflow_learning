import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import canny
import tensorflow as tf


def tf_canny(image, sigma, low_threshold, high_threshold, mask, use_quantiles):
    edge = tf.py_func(func=canny,
                      inp=[image, sigma, low_threshold, high_threshold, mask, use_quantiles],
                      Tout=tf.bool)
    return edge


if __name__ == '__main__':
    image = tf.placeholder(tf.float32, [157, 157])
    edge1 = tf_canny(image, sigma=1.0)
    edge2 = tf_canny(image, sigma=3.0)

    # Generate noisy image of a square
    im = np.zeros((128, 128))
    im[32:-32, 32:-32] = 1

    im = ndimage.rotate(im, 15, mode='constant')
    im = ndimage.gaussian_filter(im, 4)
    im += 0.2 * np.random.random(im.shape)

    with tf.Session() as sess:
        a, b = sess.run([edge1, edge2], feed_dict={image: im})
        print(a.shape)

        # display results
        plt.figure(figsize=(8, 3))

        plt.subplot(131)
        plt.imshow(im, cmap=plt.cm.jet)
        plt.axis('off')
        plt.title('noisy image', fontsize=20)

        plt.subplot(132)
        plt.imshow(a, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('Canny filter, $\sigma=1$', fontsize=20)

        plt.subplot(133)
        plt.imshow(b, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('Canny filter, $\sigma=3$', fontsize=20)

        plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)

        plt.show()
