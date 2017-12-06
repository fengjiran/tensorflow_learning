from __future__ import print_function

import platform
import numpy as np
import pandas as pd
import tensorflow as tf

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l1'

# load the train sample paths
train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
train_path = train_path[:]['image_path'].values.tolist()
print(len(train_path))
print(train_path[0:2])
# train_path = ['A.jpg', 'B.jpg', 'C.jpg']
with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(train_path, shuffle=False, capacity=64)
    reader = tf.WholeFileReader()
    name, image = reader.read(filename_queue)
    dataname = tf.train.batch([name], 2)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            print(sess.run(dataname))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
    coord.join(threads)


print('done')
