import os
import glob
import scipy
import numpy as np

from scipy.misc import imread


class Dataset_():
    """Construct dataset class."""

    def __init__(self, config, flist, augment=True):
        self.augment = augment


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, image_flist, edge_flist, mask_flist, augment=True, training=True):
        self.augment = augment
        self.training = training
        self.image_data = self.load_flist(image_flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config['INPUT_SIZE']
        self.sigma = config['SIGMA']
        self.edge = config['EDGE']
        self.mask = config['MASK']
        self.nms = config['NMS']

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config['MODE'] == 2:
            self.mask = 6

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = imread(self.data[index])

        # resize or crop if needed
        if size != 0:
            img = self.resize(img, size, size)

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

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img


if __name__ == '__main__':
    dataset = Dataset(config=1, flist='celebahq.flist', edge_flist=1, mask_flist=1)

    path = 'E:\\TensorFlow_Learning\\Inpainting\\EdgeConnect\\celebahq.flist'
    train_list = dataset.load_flist(path)
    print(len(train_list))
    print(train_list[0])
