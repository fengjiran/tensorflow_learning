# import numpy as np
import pandas as pd
import tensorflow as tf


path = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\places\\places_train_path_win.pickle'
train_path = pd.read_pickle(path)
train_path.index = range(len(train_path))
train_path = train_path[:]['image_path'].values.tolist()

with tf.device('/cpu:0'):
    for img_path in train_path:
        print(img_path)
        img_file = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img_file, channels=3)
