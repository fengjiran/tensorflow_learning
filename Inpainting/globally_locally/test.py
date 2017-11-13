import numpy as np
import tensorflow as tf
length = 10
height = 5
width = 3
x = 2
y = 4
mask = tf.pad(tf.ones([height, width]),
              paddings=[[length - height - y, y], [x, length - width - x]])
mask = tf.reshape(mask, [length, length, 1])
mask = tf.concat([mask] * 3, 2)

# mask = mask[2:8, 2:8, :]

print(mask.get_shape().as_list())

with tf.Session() as sess:
    print(sess.run(mask[:, :, 0]))
    print(sess.run(mask[:, :, 1]))
    print(sess.run(mask[:, :, 2]))

mask = np.lib.pad(np.ones([height, width]),
                  pad_width=((length - height - y, y), (x, length - width - x)),
                  mode='constant')
mask = np.reshape(mask, [length, length, 1])
mask = np.concatenate([mask] * 3, 2)

print(mask[:, :, 0])
print(mask[:, :, 1])
print(mask[:, :, 2])
