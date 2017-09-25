import tensorflow as tf

# x = tf.placeholder(tf.float32, [2, 3])
# y = tf.placeholder(tf.float32, [2, 3])

x = tf.Variable(tf.truncated_normal([2, 3]))
y = tf.Variable(tf.truncated_normal([2, 3]))

cost1 = x**2
cost2 = x**2 + y**2

grad1 = tf.gradients(ys=cost1, xs=x)
grad2 = tf.gradients(ys=cost2, xs=[x, y])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run([x, y]))
print(sess.run(grad2))

sess.close()
