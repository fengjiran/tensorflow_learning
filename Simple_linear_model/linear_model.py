from __future__ import division
from __future__ import print_function

import platform
import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import plot_images

if platform.system() == 'Windows':
    data_path = 'E:\\deeplearning_experiments\\datasets\\mnist'
elif platform.system() == 'Linux':
    data_path = '/home/richard/datasets/mnist'

data = input_data.read_data_sets(data_path, one_hot=True, validation_size=0)
data.test.cls = np.array([label.argmax() for label in data.test.labels])

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
batch_size = 128
n_epochs = 100

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.truncated_normal([img_size_flat, num_classes],
                                          stddev=1.0 / np.sqrt(float(img_size_flat))))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=y_true))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        print('Epoch: {0}/{1}'.format(epoch, n_epochs))

        for iters in range(int(len(data.train.labels) / batch_size)):
            # print('Iters: {0}/{1}'.format(iters, int(len(data.train.labels) / batch_size)))

            x_batch, y_true_batch = data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: x_batch,
                                            y_true: y_true_batch})

        acc = accuracy.eval(feed_dict={x: data.test.images,
                                       y_true: data.test.labels})

        print('Accuracy {}'.format(acc))
