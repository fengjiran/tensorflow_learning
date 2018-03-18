from __future__ import print_function

import os
from glob import glob
import platform
import numpy as np
import pandas as pd

if platform.system() == 'Windows':
    path = 'F:\\Datasets\\CelebA\\Img\\img_align_celeba_png.7z\\img_align_celeba_png'
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
elif platform.system() == 'Linux':
    path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png'
    compress_path=''
