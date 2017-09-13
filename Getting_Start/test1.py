import tensorflow as tf
import numpy as np

x_data = np.array([[161, 159, 155],
                   [189, 166, 156],
                   [212, 211, 174],
                   [173, 170, 156],
                   [217, 210, 173]]).astype(np.float32)

y_data = np.array([[377.4376, 393.5123, 361.3656],
                   [409.5895, 377.4376, 393.5123],
                   [409.5895, 393.5123, 361.3656],
                   [409.5895, 377.4376, 393.5123],
                   [393.5123, 377.4376, 409.5895]])
x_data /= 255.0
y_data /= 511.0

W2 = tf.Variable(tf.random_uniform([3, 3], -0.01, 0.01))
W1 = tf.Variable(tf.random_uniform([3, 3], -0.01, 0.01))
b = tf.Variable(tf.zeros([1, 3]))

y = tf.matmul(x_data**2, W2) + tf.matmul(x_data, W1) + b

loss = tf.reduce_mean(tf.sqrt(tf.square(y - y_data)))
optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(6000):
        sess.run(train)
        if step % 20 == 0:
            print(sess.run(loss))
