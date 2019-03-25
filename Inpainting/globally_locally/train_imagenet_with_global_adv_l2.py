from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import Conv2dLayer
from utils import BatchNormLayer
from utils import FCLayer

from utils import read_batch
# from utils import array_to_image

from models import completion_network


def global_discriminator(images, is_training, reuse=None):
    """Construct global discriminator network."""
    # batch_size = images.get_shape().as_list()[0]
    conv_layers = []
    bn_layers = []
    with tf.variable_scope('global_discriminator', reuse=reuse):
        conv1 = Conv2dLayer(images, [5, 5, 3, 64], stride=2, name='conv1')
        bn1_layer = BatchNormLayer(conv1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1_layer.output)
        conv_layers.append(conv1)
        bn_layers.append(bn1_layer)

        conv2 = Conv2dLayer(bn1, [5, 5, 64, 128], stride=2, name='conv2')
        bn2_layer = BatchNormLayer(conv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2_layer.output)
        conv_layers.append(conv2)
        bn_layers.append(bn2_layer)

        conv3 = Conv2dLayer(bn2, [5, 5, 128, 256], stride=2, name='conv3')
        bn3_layer = BatchNormLayer(conv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3_layer.output)
        conv_layers.append(conv3)
        bn_layers.append(bn3_layer)

        conv4 = Conv2dLayer(bn3, [5, 5, 256, 512], stride=2, name='conv4')
        bn4_layer = BatchNormLayer(conv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4_layer.output)
        conv_layers.append(conv4)
        bn_layers.append(bn4_layer)

        conv5 = Conv2dLayer(bn4, [5, 5, 512, 512], stride=2, name='conv5')
        bn5_layer = BatchNormLayer(conv5.output, is_training, name='bn5')
        bn5 = tf.nn.relu(bn5_layer.output)
        conv_layers.append(conv5)
        bn_layers.append(bn5_layer)

        conv6 = Conv2dLayer(bn5, [5, 5, 512, 512], stride=2, name='conv6')
        bn6_layer = BatchNormLayer(conv6.output, is_training, name='bn6')
        bn6 = tf.nn.relu(bn6_layer.output)
        conv_layers.append(conv6)
        bn_layers.append(bn6_layer)

        fc7 = FCLayer(bn6, 1, name='fc7')
        conv_layers.append(fc7)

        print('Print the global discriminator network constructure:')
        for conv_layer in conv_layers:
            tf.add_to_collection('global_dis_params_conv', conv_layer.w)
            tf.add_to_collection('global_dis_params_conv', conv_layer.b)
            tf.add_to_collection('weight_decay_global_dis', tf.nn.l2_loss(conv_layer.w))
            print('conv_{} shape:{}'.format(conv_layers.index(conv_layer) + 1, conv_layer.output_shape))

        for bn_layer in bn_layers:
            tf.add_to_collection('global_dis_params_bn', bn_layer.scale)
            tf.add_to_collection('global_dis_params_bn', bn_layer.beta)

    return fc7.output[:, 0]


if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l2'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_with_global_adv_l2'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l2'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_with_global_adv_l2'

isFirstTimeTrain = True
# input image size for comletion network and global discrimintor
input_height = 256
input_width = 256
batch_size = 32
iters_total = 500000
iters_d = 2300 * 6

weight_decay_rate = 0.0001
lr_decay_steps = 1000
init_lr = 0.001
lambda_adv = 0.0004 * 2

# placeholder
is_training = tf.placeholder(tf.bool)
global_step = tf.placeholder(tf.int64)
images = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images')
images_with_hole = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='images_with_holes')
masks_c = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 3], name='masks_c')

completed_images = completion_network(images_with_hole, is_training)

labels_G = tf.ones([batch_size])
labels_D = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)

global_dis_inputs_fake = completed_images * masks_c + images_with_hole * (1 - masks_c)

adv_pos = global_discriminator(images, is_training)
adv_neg = global_discriminator(global_dis_inputs_fake, is_training, reuse=True)
adv_all = tf.concat([adv_pos, adv_neg], axis=0)

var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
var_D = tf.get_collection('global_dis_params_conv') + tf.get_collection('global_dis_params_bn')

loss_recon = tf.reduce_mean(tf.square(masks_c * (images - completed_images)))
loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_neg,
                                                                    labels=labels_G))
loss_adv_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_all,
                                                                    labels=labels_D))

loss_G = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen')) +\
    lambda_adv * loss_adv_G
loss_D = loss_adv_D + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_global_dis') +
                                                         tf.get_collection('weight_decay_local_dis'))

lr = tf.train.exponential_decay(learning_rate=init_lr,
                                global_step=global_step,
                                decay_steps=lr_decay_steps,
                                decay_rate=0.992)

opt_g = tf.train.AdamOptimizer(learning_rate=init_lr, beta1=0.9, beta2=0.999)
opt_d = tf.train.AdamOptimizer(learning_rate=init_lr / 10, beta1=0.9, beta2=0.999)
grads_vars_g = opt_g.compute_gradients(loss=loss_G,
                                       var_list=var_G)
train_op_g = opt_g.apply_gradients(grads_vars_g)

grads_vars_d = opt_d.compute_gradients(loss=loss_D, var_list=var_D)
train_op_d = opt_d.apply_gradients(grads_vars_d)

view_g_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_g])
view_g_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_g])

view_d_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_d])
view_d_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_d])

# load the train sample paths
train_path = pd.read_pickle(compress_path)
np.random.seed(42)
train_path.index = range(len(train_path))
train_path = train_path.ix[np.random.permutation(len(train_path))]
num_batch = int(len(train_path) / batch_size)

if isFirstTimeTrain:
    old_var_G = []
    graph1 = tf.Graph()
    with graph1.as_default():
        with tf.Session(graph=graph1) as sess1:
            saver1 = tf.train.import_meta_graph(os.path.join(g_model_path, 'models_without_adv_l2.meta'))
            saver1.restore(sess1, os.path.join(g_model_path, 'models_without_adv_l2'))
            old_var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
            # print(sess1.run(tf.reduce_mean([tf.reduce_mean(a) for a in old_var_G])))
            old_var_G = sess1.run(old_var_G)
            # print(np.mean([np.mean(a) for a in old_var_G]))

saver = tf.train.Saver()
with tf.Session() as sess:
    if isFirstTimeTrain:
        sess.run(tf.global_variables_initializer())
        updates = []
        for i, item in enumerate(old_var_G):
            updates.append(tf.assign(var_G[i], item))
        sess.run(updates)
        # print(var_G[0])
        # print(sess.run(tf.reduce_mean([tf.reduce_mean(a) for a in var_G])))
        # print(len(var_G))
        # print(len(old_var_G))

        iters = 0
        with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
            pickle.dump(iters, f, protocol=2)
        saver.save(sess, os.path.join(model_path, 'models_with_global_adv_l2'))
    else:
        saver.restore(sess, os.path.join(model_path, 'models_with_global_adv_l2'))
        with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
            iters = pickle.load(f)

    while iters < iters_total:
        indx = iters % num_batch
        image_paths = train_path[indx * batch_size:(indx + 1) * batch_size]['image_path'].values
        images_, images_with_hole_, masks_c_, x_locs_, y_locs_ = read_batch(image_paths)

        if iters < iters_d:
            d_ops_list = [train_op_d, loss_D, view_d_weights, view_d_grads, view_g_weights, view_g_grads]
            # d_results_list = [_, loss_d, g_weights_mean, g_grads_mean, d_weights_mean, d_grads_mean]
            d_results_list = sess.run(d_ops_list,
                                      feed_dict={images: images_,
                                                 images_with_hole: images_with_hole_,
                                                 masks_c: masks_c_,
                                                 global_step: iters,
                                                 is_training: True})

            print('Epoch: {}, Iter: {}, loss_d: {}, d_weights_mean: {}, d_grads_mean: {}'.format(
                int(iters / num_batch) + 1,
                iters,
                d_results_list[1],
                d_results_list[2],
                d_results_list[3]
            ))
            print('    g_weights_mean: {}, g_grads_mean: {}'.format(
                d_results_list[4],
                d_results_list[5]
            ))
            iters += 1

        else:
            pass

        if iters % 100 == 0:
            with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(model_path, 'models_with_global_adv_l2'))
