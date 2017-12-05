from __future__ import print_function
import os
from glob import glob
import platform
import numpy as np
import pandas as pd

if platform.system() == 'Windows':
    path1 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\train'
    path2 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\val'
    path3 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_DET\\ILSVRC2015\\Data\\DET\\train'
    path4 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_DET\\ILSVRC2015\\Data\\DET\\val'
    path5 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\test'
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    test_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_test_path_win.pickle'
elif platform.system() == 'Linux':
    path1 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    path2 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    path3 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_DET/ILSVRC2015/Data/DET/train'
    path4 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_DET/ILSVRC2015/Data/DET/val'
    path5 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/test'
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    test_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_test_path_linux.pickle'

if not os.path.exists(compress_path):
    paths = []
    for path in (path1, path2, path3, path4):
        for filepath, _, _ in os.walk(path):
            paths.extend(glob(os.path.join(filepath, '*.JPEG')))

    paths = np.hstack(paths)
    trainset = pd.DataFrame({'image_path': paths})
    trainset.to_pickle(compress_path)

# if not os.path.exists(test_path):
#     paths = []
#     for filepath, _, _ in os.walk(path5):
#         paths.extend(glob(os.path.join(filepath, '*.JPEG')))

#     paths = np.hstack(paths)
#     testset = pd.DataFrame({'test_path': paths})
#     testset.to_pickle(test_path)
