import tensorflow as tf
import numpy as np

x = np.random.sample((100, 2))

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(9))
# dataset = dataset.batch(9)
dataset = dataset.repeat(2)
iterator = dataset.make_initializable_iterator()
el = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        print(sess.run(el))
