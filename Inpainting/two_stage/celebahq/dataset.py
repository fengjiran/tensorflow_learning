import os
import glob
import numpy as np

from .utils import create_mask


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, training=True):
        self.cfg = config
        self.training = training
        self.flist = self.cfg['FLIST']
        self.train_filenames = tf.placeholder(tf.string, shape=[None])

    def load_item(self):
        pass

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
                try:
                    print('is a file')
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []
