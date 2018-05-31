from __future__ import division
from __future__ import print_function

import os
import platform
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import array_to_image
from models import completion_network

if platform.system() == 'Windows':
    test_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\test_images'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models'
elif platform.system() == 'Linux':
    test_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/test_images'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models'

batch_size = 1

origin_image = skimage.io.imread(os.path.join(test_path, '2_origin.png')).astype(float)
image_with_hole = skimage.io.imread(os.path.join(test_path, '2_with_holes.png')).astype(float)

height = origin_image.shape[0]
width = origin_image.shape[1]

if origin_image.shape[2] == 4:
    origin_image = origin_image[:, :, 0:3]

if image_with_hole.shape[2] == 4:
    image_with_hole = image_with_hole[:, :, 0:3]

mask = np.abs(image_with_hole - origin_image)
for i in range(height):
    for j in range(width):
        if mask[i, j, 0] + mask[i, j, 1] + mask[i, j, 2] != 0:
            mask[i, j, 0] = 1.
            mask[i, j, 1] = 1.
            mask[i, j, 2] = 1.

image_with_hole /= 255.
image_with_hole = 2 * image_with_hole - 1

image_with_hole = np.reshape(image_with_hole, [1, height, width, 3])
mask = np.reshape(mask, [1, height, width, 3])

# placeholder
is_training = tf.placeholder(tf.bool)
corrupted_images = tf.placeholder(tf.float32, [batch_size, height, width, 3], name='corrupted_images')
masks_c = tf.placeholder(tf.float32, [batch_size, height, width, 3], name='masks_c')

completed_images = completion_network(corrupted_images, is_training)
test_for_show = completed_images * masks_c + corrupted_images * (1 - masks_c)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, os.path.join(model_path, 'global_local_consistent'))

    test = sess.run(test_for_show,
                    feed_dict={is_training: False,
                               corrupted_images: image_with_hole,
                               masks_c: mask})

    test = (255. * (test[0] + 1) / 2.).astype('uint8')
    plt.imshow(test)
plt.show()
# test_path = pd.read_pickle(test_path)
# np.random.seed(42)
# test_path.index = range(len(test_path))
# test_path = test_path.ix[np.random.permutation(len(test_path))]
# print(test_path[0:1]['test_path'].values)
