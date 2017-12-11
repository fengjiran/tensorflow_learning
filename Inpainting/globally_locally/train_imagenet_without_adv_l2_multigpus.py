from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2_v2'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2_v2'

isFirstTimeTrain = True
batch_size = 32
iters_c = 240000  # iters for completion network
lr_decay_steps = 1000
weight_decay_rate = 0.0001
init_lr = 0.001
num_gpus = 2


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 96
        high = 128
        image_height = 256
        image_width = 256
        gt_height = 128
        gt_width = 128

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img = tf.cast(img_decoded, tf.float32)
        img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
        img = 2 * img - 1

        hole_height, hole_width = np.random.randint(low, high, size=(2))
        y = tf.random_uniform([], 0, image_height - hole_height, tf.int32)
        x = tf.random_uniform([], 0, image_width - hole_width, tf.int32)

        mask = tf.pad(tensor=tf.ones((hole_height, hole_width)),
                      paddings=[[y, image_height - hole_height - y], [x, image_width - hole_width - x]])
        mask = tf.reshape(mask, [image_height, image_width, 1])
        mask = tf.concat([mask] * 3, 2)

        mask_1 = tf.reshape(tensor=mask[:, :, 0] * (2 * 117. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_2 = tf.reshape(tensor=mask[:, :, 1] * (2 * 104. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_3 = tf.reshape(tensor=mask[:, :, 2] * (2 * 123. / 255. - 1.),
                            shape=[image_height, image_width, 1])
        mask_tmp = tf.concat([mask_1, mask_2, mask_3], 2)

        image_with_hole = tf.identity(img)
        image_with_hole = image_with_hole * (1 - mask) + mask_tmp

        # generate the location of 128*128 patch for local discriminator
        x_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, x + hole_width - gt_width]),
                                  maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
                                  dtype=tf.int32)

        y_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, y + hole_height - gt_height]),
                                  maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
                                  dtype=tf.int32)

    return img, image_with_hole, mask, x_loc, y_loc


def _variable_on_cpu(name, shape, initializer, trainable=True):
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
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=tf.float32, trainable=trainable)

    return var


class Conv2dLayer(object):
    """Construct conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 b_init=0.,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = _variable_on_cpu(name='w',
                                      shape=filter_shape,
                                      initializer=tf.glorot_normal_initializer())

            self.b = _variable_on_cpu(name='b',
                                      shape=filter_shape[-1],
                                      initializer=tf.constant_initializer(b_init))

            linear_output = tf.nn.conv2d(self.inputs, self.w, [1, stride, stride, 1], padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DeconvLayer(object):
    """Construct deconv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 output_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = _variable_on_cpu(name='w',
                                      shape=filter_shape,
                                      initializer=tf.glorot_normal_initializer())

            self.b = _variable_on_cpu(name='b',
                                      shape=filter_shape[-2],
                                      initializer=tf.constant_initializer(0.))

            deconv = tf.nn.conv2d_transpose(value=self.inputs,
                                            filter=self.w,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding=padding)

            self.output = activation(tf.nn.bias_add(deconv, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DilatedConv2dLayer(object):
    """Construct dilated conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 rate,
                 activation=tf.identity,
                 padding='SAME',
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = _variable_on_cpu(name='w',
                                      shape=filter_shape,
                                      initializer=tf.glorot_normal_initializer())

            self.b = _variable_on_cpu(name='b',
                                      shape=filter_shape[-1],
                                      initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.atrous_conv2d(value=self.inputs,
                                                filters=self.w,
                                                rate=rate,
                                                padding=padding)

            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class BatchNormLayer(object):
    """Construct batch norm layer."""

    def __init__(self, inputs, is_training, decay=0.999, epsilon=1e-5, name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.scale = _variable_on_cpu(name='scale',
                                          shape=[inputs.get_shape()[-1]],
                                          initializer=tf.constant_initializer(1.))

            self.beta = _variable_on_cpu(name='beta',
                                         shape=[inputs.get_shape()[-1]],
                                         initializer=tf.constant_initializer(0.))

            self.pop_mean = _variable_on_cpu(name='pop_mean',
                                             shape=[inputs.get_shape()[-1]],
                                             initializer=tf.constant_initializer(0.),
                                             trainable=False)

            self.pop_var = _variable_on_cpu(name='pop_var',
                                            shape=[inputs.get_shape()[-1]],
                                            initializer=tf.constant_initializer(1.),
                                            trainable=False)

            def mean_var_update():
                axes = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs, axes)
                train_mean = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_var_update, lambda: (self.pop_mean, self.pop_var))

            self.output = tf.nn.batch_normalization(x=inputs,
                                                    mean=mean,
                                                    variance=variance,
                                                    offset=self.beta,
                                                    scale=self.scale,
                                                    variance_epsilon=epsilon)


def completion_network(images, is_training, batch_size):
    """Construct completion network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    bn_layers = []

    with tf.variable_scope('generator'):
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=1, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1_layer.output)  # N, 256, 256, 64
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [3, 3, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [3, 3, 128, 128], stride=1, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [3, 3, 128, 256], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [3, 3, 256, 256], stride=1, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        conv6 = Conv2dLayer(bn5, [3, 3, 256, 256], stride=1, name='conv6')
        bn6_layer = BatchNormLayer(conv5.output, is_training, name='bn6')
        bn6 = tf.nn.relu(bn6_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv6)
        bn_layers.append(bn6_layer)

        dilated_conv7 = DilatedConv2dLayer(bn6, [3, 3, 256, 256], rate=2, name='dilated_conv7')
        bn7_layer = BatchNormLayer(dilated_conv7.output, is_training, name='bn7')
        bn7 = tf.nn.relu(bn7_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv7)
        bn_layers.append(bn7_layer)

        dilated_conv8 = DilatedConv2dLayer(bn7, [3, 3, 256, 256], rate=4, name='dilated_conv8')
        bn8_layer = BatchNormLayer(dilated_conv8.output, is_training, name='bn8')
        bn8 = tf.nn.relu(bn8_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv8)
        bn_layers.append(bn8_layer)

        dilated_conv9 = DilatedConv2dLayer(bn8, [3, 3, 256, 256], rate=8, name='dilated_conv9')
        bn9_layer = BatchNormLayer(dilated_conv9.output, is_training, name='bn9')
        bn9 = tf.nn.relu(bn9_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv9)
        bn_layers.append(bn9_layer)

        dilated_conv10 = DilatedConv2dLayer(bn9, [3, 3, 256, 256], rate=16, name='dilated_conv10')
        bn10_layer = BatchNormLayer(dilated_conv10.output, is_training, name='bn10')
        bn10 = tf.nn.relu(bn10_layer.output)  # N, 64, 64, 256
        conv_layers.append(dilated_conv10)
        bn_layers.append(bn10_layer)

        conv11 = Conv2dLayer(bn10, [3, 3, 256, 256], stride=1, name='conv11')
        bn11_layer = BatchNormLayer(conv11.output, is_training, name='bn11')
        bn11 = tf.nn.relu(bn11_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv11)
        bn_layers.append(bn11_layer)

        conv12 = Conv2dLayer(bn11, [3, 3, 256, 256], stride=1, name='conv12')
        bn12_layer = BatchNormLayer(conv12.output, is_training, name='bn12')
        bn12 = tf.nn.relu(bn12_layer.output)  # N, 64, 64, 256
        conv_layers.append(conv12)
        bn_layers.append(bn12_layer)

        deconv13 = DeconvLayer(inputs=bn12,
                               filter_shape=[4, 4, 128, 256],
                               output_shape=[batch_size, conv2.output_shape[1],
                                             conv2.output_shape[2], 128],
                               stride=2,
                               name='deconv13')
        bn13_layer = BatchNormLayer(deconv13.output, is_training, name='bn13')
        bn13 = tf.nn.relu(bn13_layer.output)  # N, 128, 128, 128
        conv_layers.append(deconv13)
        bn_layers.append(bn13_layer)

        conv14 = Conv2dLayer(bn13, [3, 3, 128, 128], stride=1, name='conv14')
        bn14_layer = BatchNormLayer(conv14.output, is_training, name='bn14')
        bn14 = tf.nn.relu(bn14_layer.output)  # N, 128, 128, 128
        conv_layers.append(conv14)
        bn_layers.append(bn14_layer)

        deconv15 = DeconvLayer(inputs=bn14,
                               filter_shape=[4, 4, 64, 128],
                               output_shape=[batch_size, conv1.output_shape[1],
                                             conv1.output_shape[2], 64],
                               stride=2,
                               name='deconv15')
        bn15_layer = BatchNormLayer(deconv15.output, is_training, name='bn15')
        bn15 = tf.nn.relu(bn15_layer.output)  # N, 256, 256, 64
        conv_layers.append(deconv15)
        bn_layers.append(bn15_layer)

        conv16 = Conv2dLayer(bn15, [3, 3, 64, 32], stride=1, name='conv16')
        bn16_layer = BatchNormLayer(conv16.output, is_training, name='bn16')
        bn16 = tf.nn.relu(bn16_layer.output)  # N, 256, 256, 32
        conv_layers.append(conv16)
        bn_layers.append(bn16_layer)

        conv17 = Conv2dLayer(bn16, [3, 3, 32, 3], stride=1, name='conv17')
        conv_layers.append(conv17)

        print('Print the completion network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('gen_params_conv', conv_layer.w)
            tf.add_to_collection('gen_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_gen', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('gen_params_bn', bn_layer.scale)
            tf.add_to_collection('gen_params_bn', bn_layer.beta)

    return tf.nn.tanh(conv17.output)  # N, 256, 256, 3


def tower_loss(scope, images, images_with_hole, masks, split_batch_size, is_training):
    """Calculate the total loss on a single tower.

    Returns
    -------
        Tensor of shape [] containing the total loss for a batch of data.

    """
    # Build inference Graph.
    with tf.name_scope(scope):
        completed_images = completion_network(images_with_hole, is_training, split_batch_size)
        loss_recon = tf.reduce_mean(tf.square(masks * (images - completed_images)))
        loss_G = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))

    return loss_G


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    Returns
    -------
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step',
                                      [],
                                      tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        filenames = tf.placeholder(tf.string, shape=[None])
        is_training = tf.placeholder(tf.bool)

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(input_parse)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        images, images_with_hole, masks, _, _ = iterator.get_next()

        # Split the batch for towers.
        images_splits = tf.split(value=images, num_or_size_splits=num_gpus, axis=0)
        images_with_hole_splits = tf.split(value=images_with_hole, num_or_size_splits=num_gpus, axis=0)
        masks_splits = tf.split(value=masks, num_or_size_splits=num_gpus, axis=0)

        # train_path = pd.read_pickle(compress_path)
        # train_path.index = range(len(train_path))
        # train_path = train_path.ix[np.random.permutation(len(train_path))]
        # train_path = train_path[:]['image_path'].values.tolist()
        # num_batch = int(len(train_path) / batch_size)

        lr = tf.train.exponential_decay(learning_rate=init_lr,
                                        global_step=global_step,
                                        decay_steps=lr_decay_steps,
                                        decay_rate=0.992)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)

        # Get images and labels for ImageNet and split the batch across GPUs.
        assert batch_size % num_gpus == 0, ('Batch size must be divisible by number of GPUs')

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:{}'.format(i)):
                    with tf.name_scope('tower_{}'.format(i)) as scope:
                        # Calculate the loss for one tower of the model. This function
                        # constructs the entire model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope=scope,
                                          images=images_splits[i],
                                          images_with_hole=images_with_hole_splits[i],
                                          masks=masks_splits[i],
                                          split_batch_size=int(batch_size / num_gpus),
                                          is_training=is_training)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this tower.
                        grads_and_vars = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads_and_vars)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        iters = 0
        while iters < iters_c:
            sess.run([apply_gradient_op, loss])
