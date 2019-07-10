import os
import glob
import yaml
import platform as pf
import numpy as np
import cv2
from imageio import imread
from imageio import imwrite
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


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + \
                list(glob.glob(flist + '/*.JPG'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            try:
                print('is a file')
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]

    return []


if __name__ == '__main__':
    # with open('config_edge_flag.yaml', 'r') as f:
    #     cfg_flag = yaml.load(f)
    #     flag = cfg_flag['flag']
    flag = 4

    if flag == 1:
        cfg_name = 'config_edge_celeba_regular.yaml'
    elif flag == 2:
        cfg_name = 'config_edge_celeba_irregular.yaml'
    elif flag == 3:
        cfg_name = 'config_edge_celebahq_regular.yaml'
    elif flag == 4:
        cfg_name = 'config_edge_celebahq_irregular.yaml'
    elif flag == 5:
        cfg_name = 'config_edge_psv_regular.yaml'
    elif flag == 6:
        cfg_name = 'config_edge_psv_irregular.yaml'
    elif flag == 7:
        cfg_name = 'config_edge_places2_regular.yaml'
    elif flag == 8:
        cfg_name = 'config_edge_places2_irregular.yaml'

    with open(cfg_name, 'r') as f:
        cfg = yaml.load(f)

    if pf.system() == 'Windows':
        checkpoint_dir = cfg['MODEL_PATH_WIN']
        test_flist = cfg['TEST_FLIST_WIN']
        mask_flist = cfg['MASK_FLIST_WIN']
        sample_dir = cfg['SAMPLE_DIR_WIN']
    elif pf.system() == 'Linux':
        if pf.node() == 'icie-Precision-Tower-7810':
            checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
            test_flist = cfg['TEST_FLIST_LINUX_7810']
            mask_flist = cfg['MASK_FLIST_LINUX_7810']
        elif pf.node() == 'icie-Precision-T7610':
            checkpoint_dir = cfg['MODEL_PATH_LINUX_7610']
            test_flist = cfg['TEST_FLIST_LINUX_7610']
            mask_flist = cfg['MASK_FLIST_LINUX_7610']

    mask_type = 2
    # mask_paths = load_flist(mask_flist)
    # image_paths = load_flist(test_flist)

    image_paths = ['C:\\Users\\Richard\\Desktop\\img00000020.png', 'C:\\Users\\Richard\\Desktop\\img00000590.png',
                   'C:\\Users\\Richard\\Desktop\\img00000934.png', 'C:\\Users\\Richard\\Desktop\\img00001469.png']
    mask_paths = ['C:\\Users\\Richard\\Desktop\\00127_test.png', 'C:\\Users\\Richard\\Desktop\\00007_test.png',
                  'C:\\Users\\Richard\\Desktop\\00034_test.png', 'C:\\Users\\Richard\\Desktop\\00052_test.png']

    ########################### construct the model #####################################
    model = EdgeModel(cfg)
    # 1 for missing region, 0 for background
    mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    output = model.sample(gray, edge, mask)

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

        i = 0
        for img_path in image_paths:
            for mask_path in mask_paths:
                i = i + 1
                img_mask = load_mask(cfg, mask_type, mask_path)
                img_gray, img_edge = load_edge(cfg, img_path)
                feed_dict = {gray: img_gray, edge: img_edge, mask: img_mask}

                inpainted_edge = sess.run(output, feed_dict=feed_dict)
                inpainted_edge = np.reshape(inpainted_edge, [cfg['INPUT_SIZE'], cfg['INPUT_SIZE']])
                imwrite(os.path.join(sample_dir, 'test_%02d.png' % i), inpainted_edge)
