from __future__ import division
from __future__ import print_function

import platform
import numpy as np
import pandas as pd
import tensorflow as tf

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    # model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    # model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l1'


def input_parse(img_path):
    low = 96
    high = 128
    image_height = 256
    image_width = 256
    gt_height = 128
    gt_width = 128

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img = tf.cast(img_decoded, tf.float32)
    img /= 255.
    img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
    img = 2 * img - 1

    hole_height, hole_width = np.random.randint(low, high, size=(2))
    y = np.random.randint(0, image_height - hole_height)
    x = np.random.randint(0, image_width - hole_width)

    image_with_hole = tf.identity(img)
    image_with_hole[y:y + hole_height, x:x + hole_width, 0] = 2 * 117. / 255. - 1.

    return img, image_with_hole


# train_path = pd.read_pickle(compress_path)
# np.random.seed(42)
# train_path.index = range(len(train_path))
# train_path = train_path.ix[np.random.permutation(len(train_path))]
# train_path = train_path[:]['image_path'].values.tolist()


# tf.constant(train_path)
filenames = tf.constant(['E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\test_images\\a.jpg'])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
iterator = dataset.make_one_shot_iterator()
print(dataset)
print('done')

with tf.Session() as sess:
    a = sess.run(iterator.get_next())
    print(a[1].shape)
    # print(a[0][:, :, 0][0][0])
    # print(a[1][:, :, 0][0][0])
