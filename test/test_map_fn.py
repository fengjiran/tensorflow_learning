import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, [3, 2])
b = tf.placeholder(tf.float32, [3, 2])


def compute(x, y):
    return tf.reduce_mean(x), tf.reduce_mean(y)  # x + y, x - y


acc = tf.map_fn(fn=lambda x: compute(x[0], x[1]),
                elems=(a, b),
                dtype=(tf.float32, tf.float32))

with tf.Session() as sess:
    aa = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float)
    bb = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float)
    print(sess.run(acc, feed_dict={a: aa, b: bb}))
