import cv2
import tensorflow as tf
import numpy as np

from models import completion_network

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
color = (255, 255, 255)
size = 10
checkpoint_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l2\\models_without_adv_l2'
# img_path = '/Users/apple/Desktop/000013.png'
img_path = 'C:\\Users\\Richard\\Desktop\\000013.png'
batch_size = 1


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
    cv2.namedWindow('mask')
    cv2.setMouseCallback('image', erase_rect)
    cv2.setMouseCallback('mask', erase_rect)
    mask = np.zeros(img.shape)

    while True:
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img_show)
        cv2.imshow('mask', mask)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    test_img = 2 * img / 255. - 1
    test_mask = mask / 255.

    test_img = test_img * (1 - test_mask) + test_mask

    cv2.destroyAllWindows()
    # return test_img, test_mask
    return np.tile(test_img[np.newaxis, ...], [batch_size, 1, 1, 1]), np.tile(test_mask[np.newaxis, ...], [batch_size, 1, 1, 1])


def test(sess):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[0], img.shape[1]

    orig_test = 2 * img / 255. - 1
    orig_test = np.tile(orig_test[np.newaxis, ...], [batch_size, 1, 1, 1])
    orig_test = orig_test.astype(np.float32)

    test_img, test_mask = erase_img(img)
    test_img = test_img.astype(np.float32)

    print('Testing...')
    is_training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32, [batch_size, height, width, 3])
    res_image = completion_network(x, is_training, batch_size)

    saver = tf.train.Saver()
    # last_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, checkpoint_path)
    # ckpt_name = str(last_ckpt)
    # print('Loaded model file from' + ckpt_name)

    res_image = sess.run(res_image, feed_dict={x: test_img,
                                               is_training: False})
    res_image = (1 - test_mask) * orig_test + test_mask * res_image

    res = np.hstack([orig_test, test_img, res_image])
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    cv2.imshow('result', res)
    cv2.waitKey()
    print('Done.')


if __name__ == '__main__':
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(img.shape)
    # x, y = erase_img(img)
    # # print(x.shape, y.shape)
    # height, width = x.shape[1], x.shape[2]

    is_training = tf.placeholder(tf.bool)
    test_image = tf.placeholder(tf.float32, [batch_size, 218, 178, 3])
    res_image = completion_network(test_image, is_training, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        # print('Start Testing...')
        # test(sess)
