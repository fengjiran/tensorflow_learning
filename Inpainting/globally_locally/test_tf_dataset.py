import numpy as np
import tensorflow as tf

# image_string = tf.read_file('C:\\Users\\Richard\\Desktop\\image\\ILSVRC2012_test_00000003.JPEG')
# print(image_string)
# image_decoded = tf.image.decode_image(image_string)
# print(image_decoded)

# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

# iterator = dataset.make_one_shot_iterator()
# with tf.Session() as sess:
#     for i in range(5):
#         print(sess.run(iterator.get_next()))

#     print(sess.run(image_decoded).shape)
indices = tf.constant([[0], [2]])
updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]],
                       [[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]]])
shape = tf.constant([4, 4, 4])
scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
    print(sess.run(scatter))
