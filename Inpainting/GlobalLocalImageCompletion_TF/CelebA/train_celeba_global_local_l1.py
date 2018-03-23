from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import Conv2dLayer
from utils import BatchNormLayer
from utils import FCLayer
from models import completion_network

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l1'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_global_local_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_without_adv_l1'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_global_local_l1'
