from __future__ import print_function
import os
from glob import glob
import platform
import pickle
import numpy as np
import pandas as pd

if platform.system() == 'Windows':
    path1 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\train'
    path2 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_CLS_LOC\\Data\\CLS-LOC\\val'
    path3 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_DET\\ILSVRC2015\\Data\\DET\\train'
    path4 = 'F:\\Datasets\\ILSVRC2015\\ILSVRC2015_DET\\ILSVRC2015\\Data\\DET\\val'
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path.pickle'
elif platform.system() == 'Linux':
    path1 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    path2 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    path3 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_DET/ILSVRC2015/Data/DET/train'
    path4 = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_DET/ILSVRC2015/Data/DET/val'
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path.pickle'

# if not os.path.exists(compress_path):
#     paths = []
#     for path in (path1, path2, path3, path4):
#         pass
test = 'C:\\Users\\Richard\\Desktop\\image'
a = []
for filepath, _, _ in os.walk(path1):
    a.extend(glob(os.path.join(filepath, '*.JPEG')))

print(len(a))
print(a[-1])
