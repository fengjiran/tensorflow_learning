from __future__ import division
from __future__ import print_function

import tensorflow as tf


def trunc_normal():
    return tf.truncated_normal_initializer(0.0, 0.01)


def inception_v1_base(inputs, scope='InceptionV1'):
    """
    Define the Inception V1 base architecture.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to.
         can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
         'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
         'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
         'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: optional variable_scope

    Returns
    -------
    A dictionary from components of the network to the corresponding activation.

    """
    end_points = {}
    with tf.variable_scope(scope):
        end_point = 'Conv2d_1a_7x7'
        net = tf.layers.conv2d(inputs, 64, 7, strides=2, padding='same', activation=tf.nn.relu,
                               kernel_initializer=trunc_normal,
                               name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'MaxPool_2a_3x3'
        net = tf.layers.max_pooling2d(net, 3, strides=2, padding='same', name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'Conv2d_2b_1x1'
        net = tf.layers.conv2d(net, 64, 1, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_initializer=trunc_normal,
                               name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'Conv2d_2c_3x3'
        net = tf.layers.conv2d(net, 192, 3, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_initializer=trunc_normal,
                               name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'MaxPool_3a_3x3'
        net = tf.layers.max_pooling2d(net, 3, strides=2, padding='same', name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 96, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 128, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 32, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 192, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 96, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = tf.layers.max_pooling2d(net, 3, 2, padding='same', name=end_point)
        end_points[end_point] = net
        # if final_endpoint == end_point:
        #     return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 192, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 96, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 208, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 48, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 192, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 96, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 208, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 16, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 48, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 256, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 24, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 64, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 112, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 144, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 288, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 64, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 64, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 256, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 160, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 320, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 128, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'MaxPool_5a_2x2'
        net = tf.layers.max_pooling2d(net, 2, 2, padding='same', name=end_point)
        end_points[end_point] = net

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 256, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 160, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 320, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 32, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 128, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = tf.layers.conv2d(net, 384, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = tf.layers.conv2d(net, 192, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_1 = tf.layers.conv2d(branch_1, 384, 3, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.layers.conv2d(net, 48, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0a_1x1')
                branch_2 = tf.layers.conv2d(branch_2, 128, 5, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.layers.max_pooling2d(net, 3, 1, padding='same', name='MaxPool_0a_3x3')
                branch_3 = tf.layers.conv2d(branch_3, 128, 1, padding='same', activation=tf.nn.relu,
                                            kernel_initializer=trunc_normal,
                                            name='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
        end_points[end_point] = net
        return net, end_points


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.5,
                 prediction_fn=tf.nn.softmax,
                 spatial_squeeze=True,
                 global_pool=True,
                 reuse=None,
                 name='InceptionV1'):
    """Define the Inception V1 architecture.

    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    The default image size used to train this network is 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
        is_training: whether is training or not.
        dropout_keep_prob: the percentage of activation values that are retained.
        prediction_fn: a function to get predictions out of logits.
        spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
            shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.

    Returns
    -------
        net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
        end_points: a dictionary from components of the network to the corresponding
        activation.

    """
    with tf.variable_scope(name):
        net, end_points = inception_v1_base(inputs)
        with tf.variable_scope('logits'):
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                end_points['global_pool'] = net
            else:
                net = tf.layers.average_pooling2d(net, 7, strides=1)
                end_points['AvgPool_0a_7x7'] = net
            if not num_classes:
                return net, end_points
            net = tf.layers.dropout(net, dropout_keep_prob, training=is_training, name='dropout')
            logits = tf.layers.conv2d(net, num_classes, 1, name='conv2d_0c_1x1')

            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['logits'] = logits
            end_points['predictions'] = prediction_fn(logits, name='predictions')
        return logits, end_points
