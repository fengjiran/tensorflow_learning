import os
import tensorflow as tf
import numpy as np

model_path = 'E:\\TensorFlow_Learning\\Getting_Start\\my_model'

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

W1 = tf.Variable(tf.random_uniform([3, 3], -0.01, 0.01))
W2 = tf.Variable(tf.random_uniform([3, 3], -0.01, 0.01))
b = tf.Variable(tf.zeros([1, 3]))

y = tf.matmul(x_data**2, W2) + tf.matmul(x_data, W1) + b

loss = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
optimizer = tf.train.GradientDescentOptimizer(0.005)

grads = optimizer.compute_gradients(loss, [W1, W2, b])
# grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
grads = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in grads]
# grads = map(lambda gv: [tf.clip_by_value(gv[0], -1., 1.), gv[1]], grads)
train_op = optimizer.apply_gradients(grads)

# train = optimizer.minimize(loss)
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # sa = tf.train.import_meta_graph('E:\\TensorFlow_Learning\\Getting_Start\\my_model\\model.meta')
    saver.restore(sess, os.path.join(model_path, 'model'))

    for step in range(6000):
        sess.run(train_op)
        if step % 20 == 0:
            print(sess.run(loss))

    saver.save(sess, os.path.join(model_path, 'model'))
