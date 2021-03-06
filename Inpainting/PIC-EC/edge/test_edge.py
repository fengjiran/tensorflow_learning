from __future__ import print_function

import os
import yaml
import platform as pf
import numpy as np
import cv2
from scipy.misc import imread
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray
import tensorflow as tf

from networks import EdgeModel


def load_mask(cfg, mask_type=1, mask_path=None):
    if mask_type == 1:  # random block
        hole_size = cfg['INPUT_SIZE'] // 2
        top = np.random.randint(0, hole_size + 1)
        left = np.random.randint(0, hole_size + 1)
        img_mask = np.pad(array=np.ones((hole_size, hole_size)),
                          pad_width=((top, cfg['INPUT_SIZE'] - hole_size - top),
                                     (left, cfg['INPUT_SIZE'] - hole_size - left)),
                          mode='constant',
                          constant_values=0)

        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)  # (1, 256, 256, 1) float
    else:  # external
        img_mask = imread(mask_path)
        img_mask = cv2.resize(img_mask, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)
        img_mask = img_mask > 3
        img_mask = img_mask.astype(np.float32)
        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)
        img_mask = 1 - img_mask

    return img_mask  # (1, 256, 256, 1)


def load_edge(cfg, image_path):
    image = imread(image_path)  # [1024, 1024, 3], [0, 255]
    image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
    gray = rgb2gray(image)  # [256, 256], [0, 1]

    edge = canny(gray, sigma=cfg['SIGMA'])
    edge = edge.astype(np.float32)

    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    edge = np.expand_dims(edge, 0)
    edge = np.expand_dims(edge, -1)

    return gray, edge


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    if pf.system() == 'Windows':
        checkpoint_dir = cfg['MODEL_PATH_WIN']
    elif pf.system() == 'Linux':
        if pf.node() == 'icie-Precision-Tower-7810':
            checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
        elif pf.node() == 'icie-Precision-T7610':
            checkpoint_dir = cfg['MODEL_PATH_LINUX_7610']

    ############################# load the data #########################################
    mask_type = 2
    mask_path = 'F:\\Datasets\\qd_imd\\train\\00001_train.png'
    image_path = 'F:\\Datasets\\celebahq\\img00000001.png'

    img_mask = load_mask(cfg, mask_type, mask_path)
    img_gray, img_edge = load_edge(cfg, image_path)
    edge_masked = img_edge * (1 - img_mask)

    ########################### construct the model #####################################
    model = EdgeModel(cfg)
    # 1 for missing region, 0 for background
    mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    output = model.test_model(gray, edge, mask)

    feed_dict = {gray: img_gray, edge: img_edge, mask: img_mask}
    #####################################################################################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'edge_generator')
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.train.load_variable(os.path.join(checkpoint_dir, 'model'), from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')

        inpainted_edge = sess.run(output, feed_dict=feed_dict)

        plt.figure(figsize=(8, 3))

        plt.subplot(151)
        plt.imshow(img_gray[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('gray', fontsize=20)

        plt.subplot(152)
        plt.imshow(img_edge[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('edge', fontsize=20)

        plt.subplot(153)
        plt.imshow(img_mask[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('mask', fontsize=20)

        plt.subplot(154)
        plt.imshow(edge_masked[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('edge_masked', fontsize=20)

        plt.subplot(155)
        plt.imshow(inpainted_edge[0, :, :, 0], cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('inpainted', fontsize=20)

        plt.show()


#     image = imread(image_path)  # (1024, 1024, 3)
#     image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
#     img_gray = rgb2gray(image)  # (256, 256)
#     img_edge = canny(img_gray, sigma=cfg['SIGMA'])  # (256, 256)

#     img_gray = np.expand_dims(img_gray, 0)
#     img_gray = np.expand_dims(img_gray, -1)

#     img_edge = np.expand_dims(img_edge, 0)
#     img_edge = np.expand_dims(img_edge, -1)
#     img_edge = img_edge.astype(np.float)

#     feed_dict = {mask: img_mask, edge: img_edge, gray: img_gray}

#     inpainted_edge = sess.run(outputs_merged, feed_dict=feed_dict)
#     print(inpainted_edge.shape)

    # print(img_gray.shape, img_edge.shape)
