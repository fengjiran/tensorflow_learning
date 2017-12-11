from __future__ import division
from __future__ import print_function

import os
import platform
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf

from utils import Conv2dLayer
from utils import DeconvLayer
from utils import DilatedConv2dLayer
from utils import BatchNormLayer

# from models import completion_network

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2_v2'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2_v2'

isFirstTimeTrain = False
batch_size = 32
iters_c = 240000  # iters for completion network
lr_decay_steps = 1000
weight_decay_rate = 0.0001
init_lr = 0.001


def completion_network(images, is_training, batch_size):
    """Construct completion network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    bn_layers = []

    with tf.variable_scope('generator'):
        # conv_layers = []
        # bn_layers = []

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


is_training = tf.placeholder(tf.bool)
global_step = tf.get_variable('global_step',
                              [],
                              tf.int32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
# global_step = tf.placeholder(tf.int64)
filenames = tf.placeholder(tf.string, shape=[None])

dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(input_parse)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()

images, images_with_hole, masks, _, _ = iterator.get_next()
completed_images = completion_network(images_with_hole, is_training, batch_size)
var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
loss_recon = tf.reduce_mean(tf.square(masks * (images - completed_images)))
loss_G = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))

summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
# Add a summary to track the loss.
summaries.append(tf.summary.scalar('generator_loss', loss_G))

lr = tf.train.exponential_decay(learning_rate=init_lr,
                                global_step=global_step,
                                decay_steps=lr_decay_steps,
                                decay_rate=0.992)
# Add a summary to track the learning rate.
summaries.append(tf.summary.scalar('learning_rate', lr))

opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
grads_vars_g = opt.compute_gradients(loss_G, var_G)
# grads_vars_g = [(tf.clip_by_value(gv[0], -10., 10.), gv[1]) for gv in grads_vars_g]

# Add histograms for gradients.
for grad, var in grads_vars_g:
    if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

train_op_g = opt.apply_gradients(grads_vars_g, global_step)

# Add histograms for trainable variables.
for var in tf.trainable_variables():
    summaries.append(tf.summary.histogram(var.op.name, var))

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

train_op = tf.group(train_op_g, variable_averages_op)

view_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0.
                             for gv in grads_vars_g])
view_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_g])

# if isFirstTimeTrain:
#     old_var_G = []
#     graph1 = tf.Graph()
#     with graph1.as_default():
#         with tf.Session(graph=graph1) as sess1:
#             saver1 = tf.train.import_meta_graph(os.path.join(g_model_path, 'models_without_adv_l2.meta'))
#             saver1.restore(sess1, os.path.join(g_model_path, 'models_without_adv_l2'))
#             old_var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
#             old_var_G = sess1.run(old_var_G)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
summary_op = tf.summary.merge(summaries)
summary_writer = tf.summary.FileWriter(model_path)
with tf.Session() as sess:
    train_path = pd.read_pickle(compress_path)
    # np.random.seed(42)
    train_path.index = range(len(train_path))
    train_path = train_path.ix[np.random.permutation(len(train_path))]
    train_path = train_path[:]['image_path'].values.tolist()
    num_batch = int(len(train_path) / batch_size)

    sess.run(iterator.initializer, feed_dict={filenames: train_path})
    sess.run(tf.global_variables_initializer())

    if isFirstTimeTrain:
        # updates = []
        # for i, item in enumerate(old_var_G):
        #     updates.append(tf.assign(var_G[i], item))
        # sess.run(updates)

        # with open(os.path.join(g_model_path, 'iter.pickle'), 'rb') as f:
        #     iters = pickle.load(f)

        iters = 0
        with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
            pickle.dump(iters, f, protocol=2)
        saver.save(sess, os.path.join(model_path, 'models_without_adv_l2_v2'))
    else:
        saver.restore(sess, os.path.join(model_path, 'models_without_adv_l2_v2'))
        with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
            iters = pickle.load(f)

    while iters < iters_c:
        _, loss_g, weights_mean, grads_mean, gs =\
            sess.run([train_op, loss_G, view_weights, view_grads, global_step],
                     feed_dict={is_training: True})

        print('Epoch: {}, Iter: {}, loss_g: {}, weights_mean: {}, grads_mean: {}'.format(
            int(iters / num_batch) + 1,
            gs,  # iters,
            loss_g,
            weights_mean,
            grads_mean))

        iters += 1

        if iters % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict={is_training: True})
            summary_writer.add_summary(summary_str, iters)
            with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(model_path, 'models_without_adv_l2_v2'))

    # a = sess.run(iterator.get_next())
    # print(a[2].shape)
    # print(a[3], a[4])

    # plt.subplot(131)
    # plt.imshow((255. * (a[0][0] + 1) / 2.).astype('uint8'))

    # plt.subplot(132)
    # plt.imshow((255. * (a[0][1] + 1) / 2.).astype('uint8'))

    # plt.subplot(133)
    # plt.imshow((255. * (a[0][3] + 1) / 2.).astype('uint8'))

    # # plt.subplot(133)
    # # plt.imshow((255. * a[2]).astype('uint8'))

    # plt.show()
