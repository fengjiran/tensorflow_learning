import platform
import numpy as np
import pandas as pd
import tensorflow as tf
# length = 10
# height = [5, 4, 6, 3]
# width = [2, 3, 5, 4]
# x = [2, 3, 4, 2]
# y = [3, 4, 2, 4]

# for point in zip():
#     pass

# mask = tf.pad(tf.ones([height, width]),
#               paddings=[[length - height - y, y], [x, length - width - x]])
# mask = tf.reshape(mask, [length, length, 1])
# mask = tf.concat([mask] * 3, 2)

# mask = tf.reshape(mask, [1, length, length, 3])
# mask = tf.concat([mask] * 4, 0)

# print(mask.get_shape().as_list())

# with tf.Session() as sess:
#     print(sess.run(mask[2, :, :, 0]))
#     print(sess.run(mask[2, :, :, 1]))
#     print(sess.run(mask[2, :, :, 2]))
if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'

batch_size = 32

train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
train_path = train_path[:]['image_path'].values.tolist()

# index = 1598
# path = train_path[index * batch_size:(index + 1) * batch_size]

for i in range(1807854):
    print(train_path[i])

    image_contents = tf.read_file(train_path[i])
    image = tf.image.decode_image(image_contents, channels=3)

# with tf.Session() as sess:
#     init = tf.tables_initializer()
#     sess.run(init)
#     for i in range(1807854):
#         print(train_path[i])

#         image_contents = tf.read_file(train_path[i])
#         image = tf.image.decode_image(image_contents, channels=3)
#         tmp = sess.run(image)
