from __future__ import print_function

import os
from glob import glob
import platform
import numpy as np
import pandas as pd

if platform.system() == 'Windows':
    # path = 'F:\\Datasets\\CelebA\\Img\\img_align_celeba_png.7z\\img_align_celeba_png'
    path = 'F:\\Datasets\\ImageNet_100K'
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ImageNet100k\\imagenet100k_train_path_win.pickle'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        path = '/home/icie/ImageNet_100K'
        compress_path = '/home/icie/richard/MPGAN/ImageNet100k/imagenet100k_train_path_linux_7610.pickle'
    elif platform.node() == 'icie-Precision-T7810':
        path = '/home/richard/datasets/ImageNet_100K'
        compress_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ImageNet100k/imagenet100k_train_path_linux_7810.pickle'

if not os.path.exists(compress_path):
    for filepath, _, _ in os.walk(path):
        paths = glob(os.path.join(filepath, '*.png'))

    paths = np.hstack(paths)
    trainset = pd.DataFrame({'image_path': paths})
    trainset.to_pickle(compress_path)

train_path = pd.read_pickle(compress_path)

train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
train_path = train_path[:]['image_path'].values.tolist()

print(len(train_path))
# train_path.index = range(len(train_path))
