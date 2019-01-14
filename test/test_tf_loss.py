import tensorflow as tf

x = tf.placeholder(tf.float32, [2])
y = tf.placeholder(tf.float32, [2])

l1_loss = tf.losses.absolute_difference(x, y)

a = [1.0, 2.0]
b = [2.0, 3.0]

with tf.Session() as sess:
    re = sess.run(l1_loss, feed_dict={x: a, y: b})
    print(re)
