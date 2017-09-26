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
from utils import array_to_image

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

grads_vars_g = opt_g.compute_gradients(loss_G, var_G)
grads_vars_g = [(tf.clip_by_value(gv[0], -10., 10.), gv[1]) for gv in grads_vars_g]
train_op_g = opt_g.apply_gradients(grads_vars_g)

grads_vars_d = opt_d.compute_gradients(loss_D, var_D)
grads_vars_d = [(tf.clip_by_value(gv[0], -10., 10.), gv[1]) for gv in grads_vars_d]
train_op_d = opt_d.apply_gradients(grads_vars_d)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    iters = 0
    loss_D_val = 0.
    loss_G_val = 0.

    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch + 1))
        trainset.index = range(len(trainset))
        trainset = trainset.ix[np.random.permutation(len(trainset))]

        for start, end in zip(range(0, len(trainset), batch_size),
                              range(batch_size, len(trainset), batch_size)):
            index = int(start / batch_size)
            image_paths = trainset[start:end]['image_path'].values
            images_ori = map(load_image, image_paths)
            is_none = np.sum([x is None for x in images_ori])
            if is_none > 0:
                continue

            images_crops = map(crop_random, images_ori)
            train_images, train_crops, _, _ = zip(*images_crops)

            train_images = np.array(train_images)  # images with holes,the inputs of context encoder
            train_crops = np.array(train_crops)  # the holes cropped from orignal images, ground ttruth images

            if (iters != 0) and (iters % 100 == 0):
                test_image_paths = testset[:batch_size]['image_path'].values
                test_image_ori = map(load_image, test_image_paths)

                test_images_crop = [crop_random(image_ori, x=32, y=32) for image_ori in test_image_ori]
                test_images, test_crops, xs, ys = zip(*test_images_crop)

                test_images = np.array(test_images)
                test_crops = np.array(test_crops)

                recons_vals = sess.run(recons, feed_dict={
                    images: test_images,
                    ground_truth: test_crops,
                    is_training: False
                })

                recons_vals = [recons_vals[i] for i in range(recons_vals.shape[0])]

                if iters % 500 == 0:
                    ii = 0
                    for recon, img, x, y in zip(recons_vals, test_images, xs, ys):
                        recon_hid = (255. * (recon + 1) / 2.).astype('uint8')
                        test_with_crop = (255. * (img + 1) / 2.).astype('uint8')
                        test_with_crop[y:y + 64, x:x + 64, :] = recon_hid

                        test_with_crop = np.transpose(test_with_crop, [2, 0, 1])

                        image = array_to_image(test_with_crop)
                        image.save(os.path.join(result_path, 'img_' + str(ii) + '_ori.jpg'))

                        ii += 1
                        if ii > 50:
                            break

            # train generator
            sess.run(train_op_g, feed_dict={
                images: train_images,
                ground_truth: train_crops,
                is_training: True,
                learning_rate: learning_rate_val
            })

            # train discriminator
            if iters % 10 == 0:
                sess.run(train_op_d, feed_dict={})


# x = np.random.rand(batch_size, 128, 128, 3)
# y = np.random.rand(batch_size, 64, 64, 3)

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(loss_G,
#                    feed_dict={is_training: True,
#                               learning_rate: 0.001,
#                               images: x,
#                               ground_truth: y}))
