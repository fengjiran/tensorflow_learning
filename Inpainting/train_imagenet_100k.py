from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from utils import load_image
from utils import crop_random


if platform.system() == 'Windows':
    trainset_path = 'X:\\DeepLearning\\imagenet_trainset.pickle'
    testset_path = 'X:\\DeepLearning\\imagenet_testset.pickle'
    dataset_path = 'X:\\DeepLearning\\ImageNet_100K'
    result_path = 'E:\\Scholar_Project\\Inpainting\\Context_Encoders\\imagenet'
    model_path = 'E:\\Scholar_Project\\Inpainting\\Context_Encoders\\models\\imagenet'
elif platform.system() == 'Linux':
    trainset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_trainset.pickle'
    testset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_testset.pickle'
    dataset_path = '/home/richard/datasets/ImageNet_100K'
    result_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet'
    model_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/models/imagenet'
