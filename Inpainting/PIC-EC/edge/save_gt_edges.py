import os
import glob
from imageio import imread
from imageio import imwrite
from skimage.feature import canny
from skimage.color import rgb2gray
import numpy as np


def load_flist(flist):
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


img_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_images\\256'
gt_edge_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_edges'

img_list = load_flist(img_path)

i = 1
for path in img_list:
    img = imread(path)
    img_gray = rgb2gray(img)
    edge = canny(img_gray, sigma=2)
    imwrite(os.path.join(gt_edge_path, 'celebahq_gt_edge_%04d.png' % i), 1 - edge)
    i += 1
