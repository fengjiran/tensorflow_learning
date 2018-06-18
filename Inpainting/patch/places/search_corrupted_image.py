# import numpy as np
import pandas as pd
import tensorflow as tf


def input_parse(img_path):
    with tf.device('/cpu:0'):
        # print(img_path)
        img_file = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img_file, channels=3)
        # img = tf.cast(img_decoded, tf.float32)
        # img = tf.image.resize_image_with_crop_or_pad(img, cfg['img_height'], cfg['img_width'])
        # img = img / 127.5 - 1
        return img


i = 157335
path = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\places\\places_train_path_win.pickle'
train_path = pd.read_pickle(path)
num_imgs = len(train_path)
train_path.index = range(num_imgs)
train_path = train_path[:]['image_path'].values.tolist()
train_path = train_path[i:]
num_imgs = len(train_path)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.batch(1)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
batch_data = iterator.get_next()

# batch = 16
# i = 68756
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(iterator.initializer, feed_dict={filenames: train_path})
    for s in train_path:
        i += 1
        print('num: {}, path: {}'.format(i, s))
        sess.run(batch_data)
