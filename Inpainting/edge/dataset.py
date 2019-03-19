import os
import glob
import platform as pf
import yaml
import numpy as np
import tensorflow as tf
from utils import create_mask


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, training=True):
        if pf.system() == 'Windows':
            pass
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

    def __len__(self):
        """Get the length of dataset."""
        return len(self.flist)

    def load_item(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_filenames)
        train_dataset = train_dataset.map(self.input_parse)
        train_dataset = train_dataset.shuffle(buffer_size=500)
        train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.cfg['BATCH_SIZE']))
        train_dataset = train_dataset.repeat()
        # train_dataset = train_dataset.batch(self.cfg['BATCH_SIZE'], drop_remainder=True)
        train_iterator = train_dataset.make_initializable_iterator()
        images = train_iterator.get_next()
        images = tf.image.resize_area(images, [self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']])
        images = tf.clip_by_value(images, 0., 255.)
        images = images / 127.5 - 1  # [-1, 1]
        masks = create_mask(self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'],
                            self.cfg['INPUT_SIZE'] // 2, self.cfg['INPUT_SIZE'] // 2)

        return images, masks, train_iterator

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
