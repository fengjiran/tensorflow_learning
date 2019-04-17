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
from utils import tf_get_color_domain


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, flist):
        self.cfg = config
        self.flist = self.load_flist(flist)
        self.filenames = tf.placeholder(tf.string, shape=[None])
        self.iterator = None

    def __len__(self):
        """Get the length of dataset."""
        return len(self.flist)

    def load_items(self):
        images = self.load_images()
        img_grays = self.load_grayscales(images)
        img_edges = self.load_edges(img_grays)
        img_color_domains = self.load_color_domain(images)
        return images, img_edges, img_color_domains

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
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.map(self.input_parse)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.cfg['BATCH_SIZE'])
        dataset = dataset.repeat()
        # train_dataset = train_dataset.batch(self.cfg['BATCH_SIZE'], drop_remainder=True)
        self.iterator = dataset.make_initializable_iterator()
        images = self.iterator.get_next()
        images = tf.image.resize_area(images, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
        images = tf.clip_by_value(images, 0., 255.)
        images = images / 127.5 - 1  # [-1, 1]

        return images  # [N, 256, 256, 3]

    def load_grayscales(self, images):
        # images: [-1, 1]
        images = (images + 1) * 127.5  # [0, 255]
        img_grays = tf.image.rgb_to_grayscale(images)
        img_grays /= 255.  # [0, 1]
        # shape = img_grays.get_shape().as_list()
        # img_grays = tf.reshape(img_grays, [shape[0], shape[1], shape[2]])

        return img_grays  # [N, 256, 256, 1]

    def load_edges(self, img_grays, mask=None):
        sigma = self.cfg['SIGMA']
        shape = img_grays.get_shape().as_list()
        img_grays = tf.reshape(img_grays, [-1, shape[1], shape[2]])

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
            img_edges = tf.reshape(img_edges, [-1, shape[1], shape[2], 1])
            img_edges = tf.cast(img_edges, dtype=tf.float32)
            return img_edges  # [N, 256, 256, 1]

        # external
        else:
            pass

    def load_color_domain(self, images):
        images = (images + 1) * 127.5  # [0, 255]
        images = tf.cast(images, tf.uint8)
        # shape = images.get_shape().as_list()
        shape = tf.shape(images)

        blur_factor1 = self.cfg['BLUR_FACTOR1']
        blur_factor2 = self.cfg['BLUR_FACTOR2']
        k = self.cfg['K']

        img_color_domains = tf.map_fn(fn=lambda im: tf_get_color_domain(im, blur_factor1, blur_factor2, k),
                                      elems=images,
                                      dtype=tf.float32)

        img_color_domains = tf.reshape(img_color_domains, shape)
        return img_color_domains  # [N, 256, 256, 3]

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


class MaskDataset():
    """Construct mask dataset class."""

    def __init__(self, config, mask_flist):
        # if pf.system() == 'Windows':
        #     mask_flist = config['MASK_FLIST_WIN']
        # elif pf.system() == 'Linux':
        #     if pf.node() == 'icie-Precision-Tower-7810':
        #         mask_flist = config['MASK_FLIST_LINUX_7810']
        #     elif pf.node() == 'icie-Precision-T7610':
        #         mask_flist = config['MASK_FLIST_LINUX_7610']

        self.cfg = config
        self.mask_iterator = None
        self.mask_type = config['MASK']
        self.mask_flist = mask_flist

    def load_items(self):
        masks = self.load_masks()
        return masks

    def load_masks(self):

        # random block + half
        if self.mask_type == 1:
            masks = create_mask(self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'],
                                self.cfg['INPUT_SIZE'] // 2, self.cfg['INPUT_SIZE'] // 2)

            return masks  # [1, 256, 256, 1]

        # external mask
        if self.mask_type == 2:
            mask_path = tf.constant(self.load_flist(self.mask_flist))
            mask_dataset = tf.data.Dataset.from_tensor_slices(mask_path)
            mask_dataset = mask_dataset.map(self.external_mask_parse)
            mask_dataset = mask_dataset.shuffle(buffer_size=50)
            mask_dataset = mask_dataset.batch(self.cfg['BATCH_SIZE'])
            mask_dataset = mask_dataset.repeat()
            self.mask_iterator = mask_dataset.make_initializable_iterator()
            masks = self.mask_iterator.get_next()

            return masks  # [N, 256, 256, 1]

    def external_mask_parse(self, img_path):
        with tf.device('/cpu:0'):
            img_file = tf.read_file(img_path)
            img_decoded = tf.image.decode_png(img_file)  # [512, 512]
            img = tf.reshape(img_decoded, [1, 512, 512, 1])
            img = tf.image.resize_area(img, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
            img = tf.reshape(img, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'], 1])
            img = tf.cast(tf.greater(img, 3), dtype=tf.float32)
            img = tf.image.rot90(img, tf.random_uniform([], 0, 4, tf.int32))
            img = tf.image.random_flip_left_right(img)
            # img = tf.reshape(img, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'], 1])

            # 1 for the missing regions, 0 for background
            img = 1 - img

            return img  # [256, 256, 1]

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

    if pf.system() == 'Windows':
        train_flist = cfg['TRAIN_FLIST_WIN']
        val_flist = cfg['VAL_FLIST_WIN']
        test_flist = cfg['TEST_FLIST_WIN']
        mask_flist = cfg['MASK_FLIST_WIN']
    elif pf.system() == 'Linux':
        if pf.node() == 'icie-Precision-Tower-7810':
            train_flist = cfg['TRAIN_FLIST_LINUX_7810']
            val_flist = cfg['VAL_FLIST_LINUX_7810']
            test_flist = cfg['TEST_FLIST_LINUX_7810']
            mask_flist = cfg['MASK_FLIST_LINUX_7810']
        elif pf.node() == 'icie-Precision-T7610':
            train_flist = cfg['TRAIN_FLIST_LINUX_7610']
            val_flist = cfg['VAL_FLIST_LINUX_7610']
            test_flist = cfg['TEST_FLIST_LINUX_7610']
            mask_flist = cfg['MASK_FLIST_LINUX_7610']

    dataset = Dataset(cfg, val_flist)
    images, img_color_domains = dataset.load_items()
    iterator = dataset.iterator

    mask_dataset = MaskDataset(cfg, mask_flist)
    img_masks = mask_dataset.load_items()
    mask_iterator = mask_dataset.mask_iterator

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        iterators = [iterator.initializer, mask_iterator.initializer] if cfg['MASK'] == 2 else iterator.initializer

        feed_dict = {dataset.filenames: dataset.flist}

        sess.run(iterators, feed_dict=feed_dict)

        tmp0, tmp1, tmp2 = sess.run([images, img_masks, img_color_domains])

        tmp0 = (tmp0 + 1) / 2.
        print(tmp0[0].shape)
        # print(tmp2[1, :, :, 0])

        plt.figure(figsize=(8, 3))

        plt.subplot(131)
        plt.imshow(tmp0[0])
        plt.axis('off')
        plt.title('rgb', fontsize=20)

        # plt.subplot(152)
        # plt.imshow(tmp1[0, :, :, 0], cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.title('gray', fontsize=20)

        # plt.subplot(153)
        # plt.imshow(tmp2[0, :, :, 0], cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.title('edge', fontsize=20)

        plt.subplot(132)
        plt.imshow(tmp1[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('mask', fontsize=20)

        plt.subplot(133)
        plt.imshow(tmp2[0])
        plt.axis('off')
        plt.title('color_domain', fontsize=20)

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
