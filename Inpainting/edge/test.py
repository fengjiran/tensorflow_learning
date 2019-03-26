from __future__ import print_function

import os
import yaml
import platform as pf
import numpy as np
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import tensorflow as tf
import cv2
from networks import EdgeModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    checkpoint_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass

# use flist
image_path = 'img.png'
if cfg['MASK'] == 1:  # random block
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
    mask_path = None


edge_model = EdgeModel(cfg)

# 1 for missing region, 0 for background
mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])

gray_masked = gray * (1 - mask) + mask
edge_masked = edge * (1 - mask)

inputs = tf.concat([gray_masked, edge_masked, mask * tf.ones_like(gray)], axis=3)

outputs = edge_model.edge_generator(inputs)
outputs_merged = outputs * mask + edge * (1 - mask)

# saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # saver.restore(sess, os.path.join(checkpoint_dir, 'model'))
    # sess.run(tf.global_variables_initializer())
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'edge_generator')
    assign_ops = []
    for var in vars_list:
        vname = var.name
        print(vname)
        from_name = vname
        var_value = tf.train.load_variable(os.path.join(checkpoint_dir, 'model'), from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    image = imread(image_path)  # (1024, 1024, 3)
    image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
    img_gray = rgb2gray(image)  # (256, 256)
    img_edge = canny(img_gray, sigma=cfg['SIGMA'])  # (256, 256)

    img_gray = np.expand_dims(img_gray, 0)
    img_gray = np.expand_dims(img_gray, -1)

    img_edge = np.expand_dims(img_edge, 0)
    img_edge = np.expand_dims(img_edge, -1)
    img_edge = img_edge.astype(np.float)

    feed_dict = {mask: img_mask, edge: img_edge, gray: img_gray}

    inpainted_edge = sess.run(outputs_merged, feed_dict=feed_dict)
    print(inpainted_edge.shape)

    # print(img_gray.shape, img_edge.shape)
