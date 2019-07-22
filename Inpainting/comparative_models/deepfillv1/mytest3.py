import os
import argparse
from imageio import imread
from imageio import imwrite
import cv2
import numpy as np
import tensorflow as tf
# import neuralgym as ng

from inpaint_model import InpaintCAModel

model_path = 'E:\\model\\comparative_models\\deepfillv1\\celebahq'
# Image or Image folder
img_dir = 'E:\\model\\experiments\\exp3\\celebahq\\gt_images'

# Mask or Mask folder
regular_mask_dir = 'E:\\model\\experiments\\exp3\\mask\\128'
# irregular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

# Output dir
regular_output_dir = 'E:\\model\\experiments\\exp3\\celebahq\\results\\deepfillv1\\128'
# irregular_output_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\deepfillv1\\irregular'


img_list = os.listdir(img_dir)
regular_mask_list = os.listdir(regular_mask_dir)
# irregular_mask_list = os.listdir(irregular_mask_dir)


# ng.get_gpus(1)

img_tf = tf.placeholder(tf.float32, [1, 256, 256, 3])
mask_tf = tf.placeholder(tf.float32, [1, 256, 256, 1])

model = InpaintCAModel()
output = model.test(img_tf, mask_tf)
output = (output + 1.) * 127.5
output = tf.reverse(output, [-1])
output = tf.saturate_cast(output, tf.uint8)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(model_path, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    i = 1
    for dir1, dir2 in zip(img_list, regular_mask_list):
        img = imread(os.path.join(img_dir, dir1))
        mask = 255 - imread(os.path.join(regular_mask_dir, dir2))

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, -1)

        result = sess.run(output, feed_dict={img_tf: img, mask_tf: mask})
        imwrite(os.path.join(regular_output_dir, 'inpainted_128_%04d.png' % i), result[0][:, :, ::-1])
        i += 1
