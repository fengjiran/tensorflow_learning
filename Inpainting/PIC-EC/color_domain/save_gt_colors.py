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


def get_color_domain(img, blur_factor1, blur_factor2, k):  # img:[0, 255], uint8
    img_blur = cv2.medianBlur(img, blur_factor1)
    Z = img_blur.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 8
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img_blur.shape))

    img_color_domain = cv2.medianBlur(res, blur_factor2)

    img_color_domain = img_color_domain / 255.
    img_color_domain = img_color_domain.astype(np.float32)
    return img_color_domain  # [0, 1]


img_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_images\\256'
gt_color_path = 'E:\\model\\experiments\\exp4\\celebahq\\gt_colors'

img_list = load_flist(img_path)

i = 1
for path in img_list:
    img = imread(path)
    color = get_color_domain(img, 21, 3, 3)
    imwrite(os.path.join(gt_color_path, 'celebahq_gt_color_%04d.png' % i), color)
    i += 1
