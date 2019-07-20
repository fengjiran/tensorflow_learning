from imageio import imread
import numpy as np
import tensorflow as tf


def tf_psnr(img1, img2, max_val):  # for one image
    return tf.image.psnr(img1, img2, max_val)


def tf_ssim(img1, img2, max_val):  # for one image
    return tf.image.ssim(img1, img2, max_val)


def tf_l1_loss(img1, img2):  # for one image
    # convert img1 and img2 to [0,1]
    l1 = tf.losses.absolute_difference(img1, img2)
    # l1_another = tf.reduce_mean(tf.abs(img1 - img2))
    return l1


# def tf_l2_loss(img1, img2):
#     l2 = tf.losses.mean_squared_error(img1, img2)
#     return tf.reduce_mean(tf.square(batch_img1 - batch_img2))


if __name__ == '__main__':
    a = tf.placeholder(tf.float32, [1, 256, 256, 3])
    b = tf.placeholder(tf.float32, [1, 256, 256, 3])

    l1 = tf_l1_loss(a, b)
    psnr = tf_psnr(a, b, max_val=1.0)
    ssim = tf_ssim(a, b, max_val=1.0)

    with tf.Session() as sess:
        img1_path = '/Users/richard/Desktop/gt_img_65.png'
        img2_path = '/Users/richard/Desktop/img_65.png'

        img1 = imread(img1_path)
        img2 = imread(img2_path)

        img1 = np.expand_dims(img1, 0)
        img2 = np.expand_dims(img2, 0)

        img1 = img1 / 255.
        img2 = img2 / 255.

        x, y, z = sess.run([l1, psnr, ssim], feed_dict={a: img1, b: img2})
        print(x, y, z)
