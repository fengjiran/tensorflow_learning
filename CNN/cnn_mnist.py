from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import platform
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


from layers_utils import InputLayer
from layers_utils import Conv2dLayer
from layers_utils import DenseLayer
from layers_utils import FlattenLayer
from layers_utils import PoolLayer

if platform.system() == 'Windows':
    data_path = 'E:\\deeplearning_experiments\\datasets\\fashion_mnist'
elif platform.system() == 'Linux':
    data_path = '/home/richard/datasets/fashion_mnist'

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
num_channels = 1

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36

fc_size = 128

batch_size = 64


def plot_images(images, cls_true, cls_pred=None, img_shape=(28, 28)):
    """Plot 9 images in a 3x3 grid.

    Writing the true and predicted classes below each image.
    """
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_weights(weights, input_channel=0):

    with tf.Session() as sess:
        w = sess.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = np.ceil(np.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# def new_weights(shape):
#     weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#     return weights


# def new_biases(length):
#     biases = tf.Variable(tf.constant(0.05, shape=[length]))
#     return biases


# def new_conv_layer(inpt,
#                    num_input_channels,
#                    filter_size,
#                    num_filters,
#                    use_pooling=True):
#     shape = [filter_size, filter_size, num_input_channels, num_filters]
#     weights = new_weights(shape)
#     biases = new_biases(num_filters)

#     layer = tf.nn.conv2d(input=inpt,
#                          filter=weights,
#                          strides=[1, 1, 1, 1],
#                          padding='SAME')

#     layer += biases

#     if use_pooling:
#         layer = tf.nn.max_pool(value=layer,
#                                ksize=[1, 2, 2, 1],
#                                strides=[1, 2, 2, 1],
#                                padding='SAME')

#     layer = tf.nn.relu(layer)

#     return layer, weights


# def flatten_layer(layer):
#     layer_shape = layer.get_shape()
#     num_features = layer_shape[1:4].num_elements()

#     layer_flat = tf.reshape(layer, [-1, num_features])

#     return layer_flat, num_features


# def new_fc_layer(inpt,
#                  num_inputs,
#                  num_outputs,
#                  use_relu=True):
#     weights = new_weights(shape=[num_inputs, num_outputs])
#     biases = new_biases(length=num_outputs)

#     layer = tf.matmul(inpt, weights) + biases

#     if use_relu:
#         layer = tf.nn.relu(layer)

#     return layer


x = tf.placeholder(tf.float32, [None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

network = InputLayer(inputs=x_image, name='input')
print(network.outputs.get_shape().as_list())

network = Conv2dLayer(layer=network,
                      act=tf.nn.relu,
                      shape=[filter_size1, filter_size1, num_channels, num_filters1],
                      name='cnn_layer1',
                      padding='VALID')
network = PoolLayer(layer=network,
                    name='pool_layer1')
print(network.outputs.get_shape().as_list())

network = Conv2dLayer(layer=network,
                      act=tf.nn.relu,
                      shape=[filter_size2, filter_size2, num_filters1, num_filters2],
                      name='cnn_layer2',
                      padding='VALID')
network = PoolLayer(layer=network,
                    name='pool_layer2')
print(network.outputs.get_shape().as_list())

network = FlattenLayer(layer=network)
print(network.outputs.get_shape().as_list())

network = DenseLayer(layer=network,
                     n_units=fc_size,
                     act=tf.nn.relu,
                     name='dense_layer1')
print(network.outputs.get_shape().as_list())

network = DenseLayer(layer=network,
                     n_units=num_classes,
                     name='dense_layer2')
print(network.outputs.get_shape().as_list())


y_pred = tf.nn.softmax(network.outputs)
y_pred_cls = tf.argmax(y_pred, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                              labels=y_true))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        print('Epoch: {0}/{1}'.format(epoch, 100))

        for iters in range(int(len(data.train.labels) / batch_size)):
            # print('Iters: {0}/{1}'.format(iters, int(len(data.train.labels) / batch_size)))

            x_batch, y_true_batch = data.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: x_batch,
                                           y_true: y_true_batch})

        acc = accuracy.eval(feed_dict={x: data.test.images,
                                       y_true: data.test.labels})

        print('Accuracy {}'.format(acc))
