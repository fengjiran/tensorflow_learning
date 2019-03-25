import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, [3])
b = tf.placeholder(tf.float32, [3])


def compute(x, y):
    return x + y, x - y


acc = tf.map_fn(fn=lambda x: compute(x[0], x[1]),
                elems=(a, b),
                dtype=(tf.float32, tf.float32))

with tf.Session() as sess:
    print(sess.run(acc, feed_dict={a: np.array([1., 2., 3.], dtype=np.float),
                                   b: np.array([-1., 1., -1.], dtype=np.float)}))
