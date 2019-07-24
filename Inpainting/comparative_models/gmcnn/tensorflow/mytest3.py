import os
import glob
import numpy as np
from imageio import imread
from imageio import imwrite
# import subprocess
import tensorflow as tf
from options.test_options import TestOptions
# from util.util import generate_mask_rect, generate_mask_stroke
from net.network import GMCNNModel


config = TestOptions().parse()

# if os.path.isfile(config.dataset_path):
#     pathfile = open(config.dataset_path, 'rt').read().splitlines()
# elif os.path.isdir(config.dataset_path):
#     pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
# else:
#     print('Invalid testing data file/folder path.')
#     exit(1)

dataset_path = 'E:\\model\\experiments\\exp3\\celebahq\\gt_images'
# dataset_path = 'E:\\model\\experiments\\exp3\\psv\\gt_images'

regular_mask_path = 'E:\\model\\experiments\\exp3\\mask\\128'
# irregular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

# saving_path = 'E:\\model\\experiments\\exp2\\psv\\results\\gmcnn\\irregular'
# saving_path = 'E:\\model\\experiments\\exp2\\psv\\results\\gmcnn\\regular'

# saving_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\gmcnn\\irregular'
saving_path = 'E:\\model\\experiments\\exp3\\celebahq\\results\\gmcnn\\128'


model_dir = 'E:\\model\\comparative_models\\gmcnn\\celebahq'
# model_dir = 'E:\\model\\comparative_models\\gmcnn\\psv'

# pathfile = glob.glob(os.path.join(dataset_path, '*.jpg'))
pathfile = glob.glob(os.path.join(dataset_path, '*.png'))
regular_maskpath = glob.glob(os.path.join(regular_mask_path, '*.png'))
# irregular_maskpath = glob.glob(os.path.join(irregular_mask_path, '*.png'))

total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

model = GMCNNModel()

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
with tf.Session(config=sess_config) as sess:
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=reuse)
    output = (output + 1) * 127.5
    output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
    output = tf.cast(output, tf.uint8)

    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)

    for i in range(test_num):
        # mask = imread(irregular_maskpath[i])
        mask = imread(regular_maskpath[i])
        image = imread(pathfile[i])

        mask = mask / 255.
        mask = 1 - mask
        mask = np.expand_dims(mask, -1)

        # image = image * (1 - mask) + 255 * mask

        assert image.shape[:2] == mask.shape[:2]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        result = sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
        imwrite(os.path.join(saving_path, '{:04d}.png'.format(i + 1)), result[0][:, :, ::-1])
        print(' > {} / {}'.format(i + 1, test_num))
print('done.')
