from __future__ import division
from __future__ import print_function

import os
from glob import glob
import platform
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from utils import load_image
from utils import crop_random

from model import reconstruction
from model import discriminator

from loss import tf_ms_ssim
from loss import tf_l1_loss

n_epochs = 10000
learning_rate_val = 0.0003
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 128
lambda_recon = 0.9
lambda_adv = 0.1

overlap_size = 7
hiding_size = 64

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


if not os.path.exists(trainset_path) or not os.path.exists(testset_path):
    imagenet_images = []
    for filepath, _, _ in os.walk(dataset_path):
        imagenet_images.extend(glob(os.path.join(filepath, '*.JPEG')))

    imagenet_images = np.hstack(imagenet_images)

    trainset = pd.DataFrame({'image_path': imagenet_images[:int(len(imagenet_images) * 0.9)]})
    testset = pd.DataFrame({'image_path': imagenet_images[int(len(imagenet_images) * 0.9):]})

    trainset.to_pickle(trainset_path)
    testset.to_pickle(testset_path)

else:
    trainset = pd.read_pickle(trainset_path)
    testset = pd.read_pickle(testset_path)

testset.index = range(len(testset))
testset = testset.ix[np.random.permutation(len(testset))]

# placeholder
is_training = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)

images = tf.placeholder(tf.float32, [batch_size, 128, 128, 3], name='images')
ground_truth = tf.placeholder(tf.float32, [batch_size, hiding_size, hiding_size, 3], name='ground_truth')

labels_D = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)
labels_G = tf.ones([batch_size])

recons = reconstruction(images=images, is_training=is_training)

adv_pos = discriminator(images=ground_truth, is_training=is_training)
adv_neg = discriminator(images=recons, is_training=is_training, reuse=True)
adv_all = tf.concat([adv_pos, adv_neg], axis=0)

# Applying bigger loss for overlapping region
mask_recon = tf.pad(tensor=tf.ones([hiding_size - 2 * overlap_size, hiding_size - 2 * overlap_size]),
                    paddings=[[overlap_size, overlap_size], [overlap_size, overlap_size]])
