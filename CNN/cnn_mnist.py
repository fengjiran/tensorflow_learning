from __future__ import division
from __future__ import print_function

import platform
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

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
num_channels = 1


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


def new_weights(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    return weights


def new_biases(length):
    biases = tf.Variable(tf.constant(0.05, shape=[length]))
    return biases


def new_conv_layer(inpt,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape)
    biases = new_biases(num_filters)

    layer = tf.nn.conv2d(input=inpt,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(inpt,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(inpt, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
