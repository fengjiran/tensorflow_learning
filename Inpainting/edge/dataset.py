import os
import glob
import platform as pf
import yaml
# import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy import ndimage
from skimage.feature import canny
from skimage.color import rgb2gray
from utils import create_mask
from utils import tf_canny


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, training=True):
        if pf.system() == 'Windows':
            flist = config['FLIST_WIN']
        elif pf.system() == 'Linux':
            if pf.node() == 'icie-Precision-Tower-7810':
                flist = config['FLIST_LINUX_7810']
                # log_dir = cfg['LOG_DIR_LINUX_7810']
            elif pf.node() == 'icie-Precision-T7610':
                pass

        self.cfg = config
        self.training = training
        self.flist = self.load_flist(flist)
        self.train_filenames = tf.placeholder(tf.string, shape=[None])
        self.mask_filenames = tf.placeholder(tf.string, shape=[None])

    def __len__(self):
        """Get the length of dataset."""
        return len(self.flist)

    def input_parse(self, img_path):
        with tf.device('/cpu:0'):
            img_file = tf.read_file(img_path)
            img_decoded = tf.image.decode_png(img_file, channels=3)
            img = tf.cast(img_decoded, tf.float32)  # [1024, 1024, 3]
            # img = tf.image.resize_area(img, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
            # img = tf.clip_by_value(img, 0., 255.)
            # img = tf.image.resize_image_with_crop_or_pad(img, self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'])
            # img = img / 127.5 - 1
            return img  # [-1, 1]

    def load_images(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_filenames)
        train_dataset = train_dataset.map(self.input_parse)
        train_dataset = train_dataset.shuffle(buffer_size=250)
        train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.cfg['BATCH_SIZE']))
        train_dataset = train_dataset.repeat()
        # train_dataset = train_dataset.batch(self.cfg['BATCH_SIZE'], drop_remainder=True)
        train_iterator = train_dataset.make_initializable_iterator()
        images = train_iterator.get_next()
        images = tf.image.resize_area(images, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
        images = tf.clip_by_value(images, 0., 255.)
        images = images / 127.5 - 1  # [-1, 1]

        return images, train_iterator

    def load_grayscale(self, images):
        # images: [-1, 1]
        images = (images + 1) * 127.5  # [0, 255]
        img_grays = tf.image.rgb_to_grayscale(images)
        img_grays /= 255.  # [0, 1]
        # shape = img_grays.get_shape().as_list()
        # img_grays = tf.reshape(img_grays, [shape[0], shape[1], shape[2]])

        return img_grays  # [N, 256, 256, 1]

    def load_edge(self, images, mask=None):
        sigma = self.cfg['SIGMA']

        img_grays = self.load_grayscale(images)
        shape = images.get_shape().as_list()
        img_grays = tf.reshape(img_grays, [shape[0], shape[1], shape[2]])

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = tf.cast(tf.ones([shape[1], shape[2]]), dtype=tf.bool) if mask is None else tf.cast(mask, dtype=tf.bool)

        # canny
        if self.cfg['EDGE'] == 1:
            # no edge
            if sigma == -1:
                return tf.zeros([shape[1], shape[2]], dtype=tf.bool)

            # random sigma
            if sigma == 0:
                sigma = tf.random_uniform([], 1, 5)

            img_edges = tf.map_fn(fn=lambda im: tf_canny(im, sigma, mask),
                                  elems=img_grays,
                                  dtype=tf.bool)
            img_edges = tf.reshape(img_edges, [shape[0], shape[1], shape[2], 1])
            return img_edges  # [N, 256, 256, 1]

        # external
        else:
            pass

    def load_mask(self):
        # shape = images.get_shape().as_list()
        # imgh = shape[1]
        # imgw = shape[2]

        mask_type = self.cfg['MASK']

        # random block + half
        if mask_type == 1:
            masks = create_mask(self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'],
                                self.cfg['INPUT_SIZE'] // 2, self.cfg['INPUT_SIZE'] // 2)

            return masks

        # external mask
        if mask_type == 2:
            mask_dataset = tf.data.Dataset.from_tensor_slices(self.mask_filenames)
            mask_dataset = mask_dataset.map(self.external_mask_parse)
            mask_dataset = mask_dataset.shuffle(buffer_size=250)
            mask_dataset = mask_dataset.batch(self.cfg['BATCH_SIZE'])
            mask_iterator = mask_dataset.make_initializable_iterator()
            masks = mask_iterator.get_next()  # [N, 256, 256, 1]

            return masks, mask_iterator

    def external_mask_parse(self, img_path):
        with tf.device('/cpu:0'):
            img_file = tf.read_file(img_path)
            img_decoded = tf.image.decode_png(img_file)  # [512, 512]
            img = tf.reshape(img_decoded, [1, 512, 512, 1])
            img = tf.image.resize_area(img, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
            img = tf.cast(tf.greater(img, 3), dtype=tf.uint8)
            img = tf.image.rot90(img, tf.random_uniform([], 0, 4, tf.int32))
            img = tf.image.random_flip_left_right(img)

            return img  # [1, 256, 256, 1]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + \
                    list(glob.glob(flist + '/*.JPG'))
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


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    dataset = Dataset(cfg)
    images, iterator = dataset.load_images()
    grays = dataset.load_grayscale(images)
    edges = dataset.load_edge(images)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(iterator.initializer, feed_dict={dataset.train_filenames: dataset.flist})
        tmp0, tmp1, tmp2 = sess.run([images, grays, edges])
        tmp0 = (tmp0 + 1) / 2.
        print(tmp0[0].shape)

        plt.figure(figsize=(8, 3))

        plt.subplot(131)
        plt.imshow(tmp0[1])
        plt.axis('off')
        plt.title('rgb', fontsize=20)

        plt.subplot(132)
        plt.imshow(tmp1[1], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('gray', fontsize=20)

        plt.subplot(133)
        plt.imshow(tmp2[1], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('edge', fontsize=20)

        plt.show()

    # flist = dataset.load_flist(cfg['FLIST_WIN'])
    # img = imread(flist[0])
    # img_gray = rgb2gray(img)
    # print(img[:, :, 0])
    # print(img_gray)
    # img_edge = canny(img_gray, sigma=2)

    # plt.figure(figsize=(8, 3))

    # plt.subplot(131)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title('rgb', fontsize=20)

    # plt.subplot(132)
    # plt.imshow(img_gray, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.title('gray', fontsize=20)

    # plt.subplot(133)
    # plt.imshow(img_edge, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.title('edge', fontsize=20)

    # plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.02, left=0.02, right=0.98)

    # plt.show()
