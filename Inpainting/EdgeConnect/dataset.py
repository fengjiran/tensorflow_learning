import os
import glob
import numpy as np


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        self.augment = augment
        self.training = training

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


if __name__ == '__main__':
    dataset = Dataset(config=1, flist='celebahq.flist', edge_flist=1, mask_flist=1)

    path = 'E:\\TensorFlow_Learning\\Inpainting\\EdgeConnect\\celebahq.flist'
    train_list = dataset.load_flist(path)
    print(len(train_list))
    print(train_list[0])
