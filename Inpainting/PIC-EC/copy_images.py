import os
import glob
import shutil
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
    test_flist_path = 'F:\\Datasets\\flist\\celebahq_test_win.flist'
    test_flist = load_flist(test_flist_path)
    print(len(test_flist))
    print(test_flist[0])

    dst = 'E:\\model\\experiments\\exp3\\celebahq\\gt_images'

    i = 1
    for path in test_flist:
        new_img_name = 'test_img_%04d.png' % i
        shutil.copy(path, os.path.join(dst, new_img_name))
        i += 1
