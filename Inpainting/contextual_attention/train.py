from __future__ import print_function

import platform
import yaml
import numpy as np
import tensorflow as tf
from model import CompletionModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\celeba_train_path_win.pickle'
    # events_path = config['events_path_win']
    # model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\CelebA\\pretrain_model_global'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/celeba_train_path_linux.pickle'
        # events_path = config['events_path_linux']
        # model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/CelebA/pretrain_model_global'
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = '/home/icie/richard/MPGAN/CelebA/celeba_train_path_linux.pickle'
        # events_path = '/home/icie/richard/MPGAN/CelebA/models_without_adv_l1/events'
        # model_path = '/home/icie/richard/MPGAN/CelebA/pretrain_model_global'
