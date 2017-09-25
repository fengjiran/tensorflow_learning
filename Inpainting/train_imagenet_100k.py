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
alpha = 0.84

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

mask_recon = tf.reshape(mask_recon, [hiding_size, hiding_size, 1])
mask_recon = tf.concat([mask_recon] * 3, 2)
mask_overlap = 1 - mask_recon


loss_recon_center = alpha * tf_ms_ssim(recons * mask_recon, ground_truth * mask_recon, size=7, level=3) +\
    (1 - alpha) * tf_l1_loss(recons * mask_recon, ground_truth * mask_recon, size=7)

loss_recon_overlap = alpha * tf_ms_ssim(recons * mask_overlap, ground_truth * mask_overlap, size=7, level=3) +\
    (1 - alpha) * tf_l1_loss(recons * mask_overlap, ground_truth * mask_overlap, size=7)

loss_recon = loss_recon_center / 10. + loss_recon_overlap


loss_adv_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_all,
                                                                    labels=labels_D))
loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_neg,
                                                                    labels=labels_G))
loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
loss_D = loss_adv_D * lambda_adv

# Trainable variables in generator and discriminator
var_G = filter(lambda x: x.name.startswith('generator'), tf.trainable_variables())
var_D = filter(lambda x: x.name.startswith('discriminator'), tf.trainable_variables())

w_G = filter(lambda x: x.name.endswith('w:0'), var_G)
w_D = filter(lambda x: x.name.endswith('w:0'), var_D)

# loss_G = loss_G + weight_decay_rate * tf.reduce_mean(tf.stack(list(map(tf.nn.l2_loss, w_G))))
# loss_D = loss_D + weight_decay_rate * tf.reduce_mean(tf.stack(list(map(tf.nn.l2_loss, w_D))))

loss_G = loss_G + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))
loss_D = loss_D + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_dis'))

opt_g = tf.train.AdamOptimizer(learning_rate)
opt_d = tf.train.AdamOptimizer(learning_rate)

x = np.random.rand(batch_size, 128, 128, 3)
y = np.random.rand(batch_size, 64, 64, 3)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss_G,
                   feed_dict={is_training: True,
                              learning_rate: 0.001,
                              images: x,
                              ground_truth: y}))
