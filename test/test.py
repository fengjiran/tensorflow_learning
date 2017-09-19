import tensorflow as tf
from tensorflow.python.client import device_lib

# g = tf.get_default_graph()

# x = tf.Variable(1.0, name='x')
# x_plus_1 = tf.assign_add(x, 1)
# # y = x
# # y = tf.identity(x)
# # y = x_plus_1

# with tf.control_dependencies([x_plus_1]):
#     # z = tf.identity(x)
#     # y = x
#     y = tf.identity(x)
# # y = tf.identity(x)
# # y = x_plus_1
# # y = tf.identity(x)
# # z = tf.identity(x, name='x')
# # print(g.get_operations())
# print(tf.global_variables())
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(5):
#         # print(sess.run(x), sess.run(y), sess.run(x_plus_1))
#         # print(sess.run(x_plus_1), sess.run(x), sess.run(y))
#         # print(sess.run(z))
#         print(sess.run(y))


# g1 = tf.get_default_graph()
# c1 = tf.constant(4.0)
# assert c1.graph is g1

# print(g1)

# g2 = tf.Graph()
# with g2.as_default():
#     # Define operations and tensors in `g`.
#     c2 = tf.constant(30.0)
#     assert c2.graph is g2
# print(g2)
# print(device_lib.list_local_devices())
# print(g2.get_operations())

pred = tf.placeholder(tf.bool)
x = tf.Variable([1])
# assign_x_2 = tf.assign(x, [2])


def update_x_2():
    with tf.control_dependencies([tf.assign(x, [2])]):
        return tf.identity(x)


y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y, feed_dict={pred: False}))
    print(sess.run(y, feed_dict={pred: True}))
