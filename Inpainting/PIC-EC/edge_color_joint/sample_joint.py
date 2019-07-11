import os
import glob
import yaml
import numpy as np
import cv2
from imageio import imread
from imageio import imwrite
import tensorflow as tf

from utils import get_color_domain
from networks import InpaintModel


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

    return img_mask  # (1, 256, 256, 1) float


def load_color(cfg, image_path):
    image = imread(image_path)  # [1024, 1024, 3], [0, 255]
    image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
    color = get_color_domain(image, cfg['BLUR_FACTOR1'], cfg['BLUR_FACTOR1'], cfg['K'])

    image = np.expand_dims(image, 0)  # (1, 256, 256, 3)
    color = np.expand_dims(color, 0)  # (1, 256, 256, 3)

    image = image / 127.5 - 1

    return image, color


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
    with open('test_joint_flag.yaml', 'r') as f:
        cfg_flag = yaml.load(f, Loader=yaml.FullLoader)
        flag = cfg_flag['flag']

    if flag == 1:
        cfg_name = 'test_joint_celeba_regular.yaml'
    elif flag == 2:
        cfg_name = 'test_joint_celeba_irregular.yaml'
    elif flag == 3:
        cfg_name = 'test_joint_celebahq_regular.yaml'
    elif flag == 4:
        cfg_name = 'test_joint_celebahq_irregular.yaml'
    elif flag == 5:
        cfg_name = 'test_joint_psv_regular.yaml'
    elif flag == 6:
        cfg_name = 'test_joint_psv_irregular.yaml'
    elif flag == 7:
        cfg_name = 'test_joint_places2_regular.yaml'
    elif flag == 8:
        cfg_name = 'test_joint_places2_irregular.yaml'

    with open(cfg_name, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint_dir = cfg['MODEL_PATH']
    sample_dir = cfg['SAMPLE_DIR']
    mask_type = cfg['MASK']
    mask_paths = load_flist(cfg['TEST_MASK_PATH'])
    image_paths = load_flist(cfg['TEST_IMAGE_PATH'])

    ########################### construct the model ##################################
    model = ColorModel(cfg)
    # 1 for missing region, 0 for background
    mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
    image = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
    color = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
    output = model.test_model(image, color, mask)
    ##################################################################################

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'color_generator')
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
                img, img_color = load_color(cfg, img_path)
                feed_dict = {image: img, color: img_color, mask: img_mask}

                inpainted_color = sess.run(output, feed_dict=feed_dict)
                inpainted_color = np.reshape(inpainted_color, [cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
                imwrite(os.path.join(sample_dir, 'test_%02d.png' % i), inpainted_color)
