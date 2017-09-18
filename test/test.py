import tensorflow as tf
from tensorflow.python.client import device_lib

x = tf.Variable(1.0)
x_plus_1 = tf.assign_add(x, 1)
# y = tf.identity(x)

with tf.control_dependencies([x_plus_1]):
    print('hello')
    # y = x_plus_1
    # y = tf.identity(x)
    # z = tf.identity(x, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        # print(sess.run(x), sess.run(y), sess.run(x_plus_1))
        # print(sess.run(x_plus_1), sess.run(x), sess.run(y))
        print(sess.run(x_plus_1))


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
