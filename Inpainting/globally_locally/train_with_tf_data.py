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
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_file, img_decoded


train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
train_path = train_path[:]['image_path'].values.tolist()

filenames = tf.constant(train_path)
dataset = tf.data.Dataset.from_tensor_slices(filenames)

dataset = dataset.map(input_parse)
print(dataset)
print('done')
