import os
import yaml
import platform as pf
import numpy as np
import tensorflow as tf
import cv2
from .networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    checkpoint_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        checkpoint_dir = cfg['MODEL_PATH_LINUX_7610']


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
