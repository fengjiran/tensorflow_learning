import os
import glob
from imageio import imread
from imageio import imwrite
import numpy as np
import cv2


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


src_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_images\\1024'
dst_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_images\\256'

img_list = load_flist(src_path)

i = 1
for path in img_list:
    img = imread(path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    imwrite(os.path.join(dst_path, 'gt_img_256_%04d.png' % i), img)
    i += 1
