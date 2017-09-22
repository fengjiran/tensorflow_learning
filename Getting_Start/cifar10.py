# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

# Compute input images and labels for training. If you would like to run
# evaluations, use inputs() instead.
inputs, labels = distorted_inputs()

# Compute inference on the model inputs to make a prediction.
predictions = inference(inputs)

# Compute the total loss of the prediction with respect to the labels.
loss = loss(predictions, labels)

# Create a graph to run one step of training with respect to the loss.
train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _variable_on_cpu(name, shape, initializer):
    """Help to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns
    -------
      Variable Tensor

    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var


def _variable_with_decay(name, shape, stddev, wd):
    """Help to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2 Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns
    -------
      Variable Tensor

    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def _activation_summary(x):
    """Help to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor

    Returns
    -------
      nothing

    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + 'sparsity', tf.nn.zero_fraction(x))


def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns
    -------
      Logits.

    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_decay(name='weights',
                                      shape=[5, 5, 3, 64],
                                      stddev=5e-2,
                                      wd=0.0)

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv1
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_decay(name='weights',
                                      shape=[5, 5, 64, 64],
                                      stddev=5e-2,
                                      wd=0.0)

        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value

        weights = _variable_with_decay('weights',
                                       shape=[dim, 384],
                                       stddev=0.04,
                                       wd=0.004)

        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_decay('weights',
                                       shape=[384, 192],
                                       stddev=0.04,
                                       wd=0.004)

        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_decay('weights',
                                       shape=[192, 10],
                                       stddev=1 / 192.0,
                                       wd=0.0)
        biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns
    -------
      Loss tensor of type float.

    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# https://zhuanlan.zhihu.com/p/27017189


def pre_process_image(image, training):
    """Take a single image as input to pre-process.

    A boolean whether to build the training or testing graph.
    """
    if training:
        # For training, add the following to the tensorflow graph
        # Randomly crop the input image
        image = tf.random_crop(image, size=[24, 24, 3])

        # Randomly flip the image horizontally
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue,contrast and saturation
        image = tf.image.random_hue(image, max_delta=0.05)

        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)

        image = tf.image.random_brightness(image, max_delta=0.2)

        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    else:
        # Crop the input image around the center so it is the same
        # size as images that are randomly cropped during training
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=24,
                                                       target_width=24)

    return image


def pre_process(images, training):
    """Use tensorflow to loop over all the input images.

    Call the function above which takes a single image as
    input.
    """
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images
