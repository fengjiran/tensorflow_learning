from __future__ import print_function
from __future__ import division

import tensorflow as tf


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          global_pool=True,
          name='vgg_a'):
    """Construct a example vgg 11-layers version network.

    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

    Returns
    -------
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.

    """
    end_points = {}
    with tf.variable_scope(name):
        # Block1
        end_point = 'conv1'
        x = tf.layers.conv2d(inputs, 64, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool1'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block2
        end_point = 'conv2'
        x = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool2'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block3
        end_point = 'conv3'
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv4'
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool3'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block4
        end_point = 'conv5'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv6'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool4'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block5
        end_point = 'conv7'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv8'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool5'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Use conv2d instead of fully connected layers.
        x = tf.layers.conv2d(x, 4096, 7, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name='fc9')
        x = tf.layers.dropout(x, dropout_keep_prob, training=is_training, name='dropout6')

        x = tf.layers.conv2d(x, 4096, 1, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name='fc10')
        end_points['fc'] = x

        if global_pool:
            x = tf.reduce_mean(x, [1, 2], keepdims=True, name='global_pool')
            end_points['global_pool'] = x
        if num_classes:
            x = tf.layers.dropout(x, dropout_keep_prob, training=is_training,
                                  name='dropout7')
            x = tf.layers.conv2d(x, num_classes, 1,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                 name='fc11')

        if spatial_squeeze:
            x = tf.squeeze(x, name='fc11/squeezed')
            end_points['squeezed'] = x
    return x, end_points


def vgg16(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          global_pool=True,
          name='vgg_a'):
    """Construct a example vgg 11-layers version network.

    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

    Returns
    -------
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.

    """
    end_points = {}
    with tf.variable_scope(name):
        # Block1
        end_point = 'conv1'
        x = tf.layers.conv2d(inputs, 64, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x
        print(x.get_shape())

        end_point = 'pool1'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x
        print(x.get_shape())

        # Block2
        end_point = 'conv2'
        x = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool2'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x
        print(x.get_shape())

        # Block3
        end_point = 'conv3'
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv4'
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool3'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block4
        end_point = 'conv5'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv6'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool4'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x

        # Block5
        end_point = 'conv7'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'conv8'
        x = tf.layers.conv2d(x, 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name=end_point)
        end_points[end_point] = x

        end_point = 'pool5'
        x = tf.layers.max_pooling2d(x, 2, 2, padding='same', name=end_point)
        end_points[end_point] = x
        print(x.get_shape())

        # Use conv2d instead of fully connected layers.
        x = tf.layers.conv2d(x, 4096, 7, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name='fc9')
        x = tf.layers.dropout(x, dropout_keep_prob, training=is_training, name='dropout6')

        x = tf.layers.conv2d(x, 4096, 1, activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                             name='fc10')
        end_points['fc'] = x

        if global_pool:
            x = tf.reduce_mean(x, [1, 2], keepdims=True, name='global_pool')
            end_points['global_pool'] = x
        if num_classes:
            x = tf.layers.dropout(x, dropout_keep_prob, training=is_training,
                                  name='dropout7')
            x = tf.layers.conv2d(x, num_classes, 1,
                                 kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                 name='fc11')

        if spatial_squeeze:
            x = tf.squeeze(x, name='fc11/squeezed')
            end_points['squeezed'] = x
    return x, end_points


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [10, 224, 224, 3])
    outputs, end_points = vgg_a(inputs)
    print(outputs.get_shape())
