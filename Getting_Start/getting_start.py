# import os
# from importlib import reload
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from keras import backend as K


# def set_keras_backend(backend):
#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend


# set_keras_backend('tensorflow')
# print(K.backend())
# print('hello world')
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)

print(result)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_triple = adder_node * 3.

with tf.Session() as sess:
    print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))
    print(sess.run(add_and_triple, feed_dict={a: [1, 3], b: [2, 4]}))

W = tf.Variable(initial_value=[.3], dtype=tf.float32)
b = tf.Variable(initial_value=[-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # sess.run([fixW, fixb])
    for i in range(1000):
        sess.run(train, feed_dict={x: [1, 2, 3, 4],
                                   y: [0, -1, -2, -3]})
    print(sess.run([W, b]))
