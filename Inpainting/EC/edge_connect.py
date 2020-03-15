import os
import glob
import numpy as np
import tensorflow as tf
from networks import InpaintingModel


class EdgeConnenct():
    """Construct edge connect model."""

    def __init__(self, config=None):
        self.cfg = config

        if self.cfg['MODEL'] == 1:
            model_name = 'edge'
        elif self.cfg['MODEL'] == 2:
            model_name = 'inpaint'
        elif self.cfg['MODEL'] == 3:
            model_name = 'edge_inpaint'
        elif self.cfg['MODEL'] == 4:
            model_name = 'joint'

    def train(self):
        epoch = 0
        keep_training = True
        model = self.cfg['MODEL']
        max_iteration = int(self.cfg['MAX_ITERS'])

    def load_trainset(self):
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
