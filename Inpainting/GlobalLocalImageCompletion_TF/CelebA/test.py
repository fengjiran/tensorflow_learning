import cv2
import tensorflow as tf
import numpy as np

from models import completion_network

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
color = (255, 255, 255)
size = 10
checkpoint_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l2'
# img_path = '/Users/apple/Desktop/000013.png'
img_path = 'C:\\Users\\Richard\\Desktop\\000013.png'
batch_size = 32


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
        # img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img)
        # cv2.imshow('mask', mask)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    # test_img = cv2.resize(img, (178, 178))
    # test_img = 2 * test_img / 255. - 1
    # test_mask = cv2.resize(mask, (178, 178)) / 255.
    test_img = 2 * img / 255. - 1
    test_mask = mask / 255.

    test_img = test_img * (1 - test_mask) + test_mask

    cv2.destroyAllWindows()
    return test_img, test_img
    # return np.tile(test_img[np.newaxis, ...], [batch_size, 1, 1, 1]), np.tile(test_mask[np.newaxis, ...], [batch_size, 1, 1, 1])


def test(sess, model):
    saver = tf.train.Saver()
    last_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)

    print('Loaded model file from' + ckpt_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    orig_test = cv2.resize(img, (178, 178)) / 127.5 - 1
    orig_test = np.tile(orig_test[np.newaxis, ...], [batch_size, 1, 1, 1])
    orig_test = orig_test.astype(np.float32)

    orig_h, orig_w = img.shape[0], img.shape[1]

    test_img, test_mask = erase_img(img)
    test_img = test_img.astype(np.float32)

    print('Testing...')


if __name__ == '__main__':
    img = cv2.imread(img_path)
    print(img.shape)
    x, y = erase_img(img)
    # print(x.shape, y.shape)
    height, width = x.shape[0], x.shape[1]

    is_training = tf.placeholder(tf.bool)
    test_image = tf.placeholder(tf.float32, [1, height, width, 3])
    res_image = completion_network(test_image, is_training, 1)
    print(res_image.get_shape())
