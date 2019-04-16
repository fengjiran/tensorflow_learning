import os
import yaml
import platform as pf
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import cv2
from networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    checkpoint_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7610']


def load_mask(mask_path=None):
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
        img_mask = imread(mask_path)
        img_mask = cv2.resize(img_mask, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)
        img_mask = np.expand_dims(img_mask, 0)
        img_mask = img_mask.astype(np.float32)
        img_mask = 1 - img_mask

    return img_mask


if __name__ == '__main__':
    model = ColorModel(cfg)

    image = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
    color_domain = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
    mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])

    output = model.test_model(image, color_domain, mask)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'color_generator')
        assign_ops = []
        for var in vars_list:
            vname = var.name
            print(vname)
            from_name = vname
            var_value = tf.train.load_variable(os.path.join(checkpoint_dir, 'model'), from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
