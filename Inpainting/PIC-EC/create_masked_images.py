import os
import glob
from imageio import imread
from imageio import imwrite
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


if __name__ == '__main__':
    gt_img_dir = 'E:\\model\\experiments\\exp2\\psv\\gt_images'
    mask_dir = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

    result_dir = 'E:\\model\\experiments\\exp2\\psv\\irregular_masked_imgs'

    gt_img_list = load_flist(gt_img_dir)
    mask_list = load_flist(mask_dir)

    i = 1
    for path1, path2 in zip(gt_img_list, mask_list):
        img = imread(path1)
        mask = imread(path2)

        img = img / 255.
        mask = mask / 255.

        mask = np.expand_dims(mask, -1)

        masked_img = img * mask + 1 - mask

        imwrite(os.path.join(result_dir, 'irregular_masked_img_256_%03d.png' % i), masked_img)
        i += 1
