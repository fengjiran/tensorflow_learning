from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import BatchNormLayer
from utils import FCLayer

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\models_without_adv_l1'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\models_global_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/models_without_adv_l1'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/models_global_l1'

# isFirstTimeTrain = False
isFirstTimeTrain = True
batch_size = 16
weight_decay_rate = 1e-4
init_lr_g = 3e-4
init_lr_d = 3e-4
lr_decay_steps = 1000
iters_total = 100000
iters_d = 10000
alpha_rec = 0.995
alpha_global = 0.0025
alpha_local = 0.0025

gt_height = 96
gt_width = 96
