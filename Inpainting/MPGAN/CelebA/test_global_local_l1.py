from __future__ import print_function

import platform
import cv2
import tensorflow as tf
import numpy as np

from mpgan_models import completion_network

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
color = (255, 255, 255)
size = 25
batch_size = 1

if platform.system() == 'Windows':
    checkpoint_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\models_global_local_l1\\models_global_local_l1'
    img_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\000013.png'

elif platform.system() == 'Linux':
    checkpoint_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/models_global_local_l1/models_global_local_l1'
    img_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/000013.png'

elif platform.system() == 'Darwin':
    # checkpoint_path = '/Users/apple/Desktop/richard/Tensorflow_Learning/Inpainting/MPGAN/CelebA/models_global_local_l1/models_global_local_l1'
    checkpoint_path = '/Users/apple/Desktop/richard/Tensorflow_Learning/Inpainting/MPGAN/CelebA/models_global_l1/models_global_l1'
    img_path = '/Users/apple/Desktop/richard/Tensorflow_Learning/Inpainting/MPGAN/CelebA/000013.png'


def erase_img(img):
    # mouse callback function
    def erase_rect(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
            cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), color, -1)

    cv2.namedWindow('image')
    # cv2.namedWindow('mask')
    cv2.setMouseCallback('image', erase_rect)
    cv2.setMouseCallback('mask', erase_rect)
    mask = np.zeros(img.shape)

    while True:
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img_show)
        # cv2.imshow('mask', mask)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # enter
            break

    test_img = img / 127.5 - 1
    test_mask = mask / 255.

    test_img = test_img * (1 - test_mask) + test_mask

    cv2.destroyAllWindows()
    return np.tile(test_img[np.newaxis, ...], [batch_size, 1, 1, 1]), np.tile(test_mask[np.newaxis, ...], [batch_size, 1, 1, 1])


def test(sess):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[0], img.shape[1]

    orig_test = img / 127.5 - 1
    orig_test = np.tile(orig_test[np.newaxis, ...], [batch_size, 1, 1, 1])
    orig_test = orig_test.astype(np.float32)

    test_img, test_mask = erase_img(img)
    test_img = test_img.astype(np.float32)

    print('Testing...')
    is_training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32, [batch_size, height, width, 3])
    res_image = completion_network(x, is_training, batch_size)
    variable_averages = tf.train.ExponentialMovingAverage(decay=0.999)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_path)
    # sess.run(tf.global_variables_initializer())

    res_image = sess.run(res_image, feed_dict={x: test_img,
                                               is_training: False})

    res_image = (1 - test_mask) * orig_test + test_mask * res_image
    res_image = res_image.astype(np.float32)

    orig = (orig_test[0] + 1) / 2
    test = (test_img[0] + 1) / 2
    recon = (res_image[0] + 1) / 2

    res = np.hstack([orig, test, recon])
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    cv2.imshow('result', res)
    cv2.waitKey()
    print('Done.')


if __name__ == '__main__':
    with tf.Session() as sess:
        print('Start Testing...')
        test(sess)
