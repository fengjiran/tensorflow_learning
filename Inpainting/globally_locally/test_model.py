from __future__ import division
from __future__ import print_function

import os
import platform
import pandas as pd
import numpy as np
import tensorflow as tf

from utils import read_batch
from utils import array_to_image

from models import completion_network
from models import combine_discriminator

if platform.system() == 'Windows':
    test_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_test_path_win.pickle'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models'
elif platform.system() == 'Linux':
    test_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_test_path_linux.pickle'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models'

batch_size = 1

# placeholder
is_training = tf.placeholder(tf.bool)
images = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='images_with_holes')
masks_c = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='masks_c')

completed_images = completion_network(images_with_hole, is_training)
test_for_show = completed_images * masks_c + images_with_hole * (1 - masks_c)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, os.path.join(model_path, 'global_local_consistent'))

    sess.run(test_for_show,
             feed_dict={is_training: False})

# test_path = pd.read_pickle(test_path)
# np.random.seed(42)
# test_path.index = range(len(test_path))
# test_path = test_path.ix[np.random.permutation(len(test_path))]
# print(test_path[0:1]['test_path'].values)
