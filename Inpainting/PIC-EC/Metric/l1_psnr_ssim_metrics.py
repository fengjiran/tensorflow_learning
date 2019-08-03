import os
import glob
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


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            try:
                print('is a file')
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]

    return []
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
        fake_dir = 'E:\\model\\experiments\\exp3\\psv\\results\\PIC-EC\\128'
        real_dir = 'E:\\model\\experiments\\exp3\\psv\\gt_images'

        real_flist = load_flist(real_dir)
        fake_flist = load_flist(fake_dir)

        l1_list = []
        psnr_list = []
        ssim_list = []

        i = 1
        for real_img_path, fake_img_path in zip(real_flist, fake_flist):
            real_img = imread(real_img_path)
            fake_img = imread(fake_img_path)

            real_img = np.expand_dims(real_img, 0)
            fake_img = np.expand_dims(fake_img, 0)

            real_img = real_img / 255.
            fake_img = fake_img / 255.

            x, y, z = sess.run([l1, psnr, ssim], feed_dict={a: real_img, b: fake_img})
            print('i:{}, l1:{}, psnr:{}, ssim:{}'.format(i, x, y, z))

            l1_list.append(x)
            psnr_list.append(y)
            ssim_list.append(z)

            i += 1

        print('(l1, std):({} {})'.format(np.mean(l1_list), np.std(l1_list)))
        # print('l1:{}, std:{}'.format(np.mean(l1_list), np.std(l1_list)))
        print('(psnr, std):({} {})'.format(np.mean(psnr_list), np.std(psnr_list)))
        print('(ssim, std):({} {})'.format(np.mean(ssim_list), np.std(ssim_list)))
