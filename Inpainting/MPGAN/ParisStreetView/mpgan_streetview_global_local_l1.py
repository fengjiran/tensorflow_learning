from __future__ import division
from __future__ import print_function

import os
import pickle
import platform
import numpy as np
import pandas as pd
import tensorflow as tf


from mpgan_models import completion_network
from mpgan_models import global_discriminator
from mpgan_models import markovian_discriminator

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ParisStreetView\\parisstreetview_train_path_win.pickle'
    g_model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ParisStreetView\\models_without_adv_l1'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\MPGAN\\ParisStreetView\\models_global_local_l1'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-Tower-7810':
        compress_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ParisStreetView/parisstreetview_train_path_linux_7810.pickle'
        g_model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ParisStreetView/models_without_adv_l1'
        model_path = '/home/richard/TensorFlow_Learning/Inpainting/MPGAN/ParisStreetView/models_global_local_l1'
    elif platform.node() == 'icie-Precision-T7610':
        compress_path = '/home/icie/richard/MPGAN/ParisStreetView/parisstreetview_train_path_linux_7610.pickle'
        g_model_path = '/home/icie/richard/MPGAN/ParisStreetView/models_without_adv_l1'
        model_path = '/home/icie/richard/MPGAN/ParisStreetView/models_global_local_l1'

isFirstTimeTrain = False
# isFirstTimeTrain = True
# isFirstTimeTrain_G = True
isFirstTimeTrain_Joint = True
batch_size = 2
weight_decay_rate = 1e-4
init_lr_g = 3e-4
init_lr_d = 3e-5
lr_decay_steps = 1000
iters_c = 5 * int(14900 / batch_size)
iters_total = 200000
iters_d = 15000
alpha_rec = 0.8
alpha_global = 0.1
alpha_local = 0.1

gt_height = 256
gt_width = 256


def input_parse(img_path):
    with tf.device('/cpu:0'):
        low = 192
        high = 256
        image_height = 500
        image_width = 500
        gt_height = 256
        gt_width = 256

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        img = tf.cast(img_decoded, tf.float32)
        # img /= 255.
        img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)

        # input image range from -1 to 1
        # img = 2 * img - 1
        img = img / 127.5 - 1

        ori_image = tf.identity(img)

        hole_height, hole_width = np.random.randint(low, high, size=(2))
        y = tf.random_uniform([], 0, image_height - hole_height, tf.int32)
        x = tf.random_uniform([], 0, image_width - hole_width, tf.int32)

        mask = tf.pad(tensor=tf.ones((hole_height, hole_width)),
                      paddings=[[y, image_height - hole_height - y],
                                [x, image_width - hole_width - x]])
        mask = tf.reshape(mask, [image_height, image_width, 1])
        mask = tf.concat([mask] * 3, 2)

        image_with_hole = tf.multiply(img, 1 - mask) + mask

        # generate the location of 300*300 patch for local discriminator
        x_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, x + hole_width - gt_width]),
                                  maxval=tf.reduce_min([x, image_width - gt_width]) + 1,
                                  dtype=tf.int32)
        y_loc = tf.random_uniform(shape=[],
                                  minval=tf.reduce_max([0, y + hole_height - gt_height]),
                                  maxval=tf.reduce_min([y, image_height - gt_height]) + 1,
                                  dtype=tf.int32)

        hole_height = tf.convert_to_tensor(hole_height, tf.float32)
        hole_width = tf.convert_to_tensor(hole_width, tf.float32)
        return ori_image, image_with_hole, mask, x_loc, y_loc, hole_height, hole_width


def train():
    # is_training = tf.placeholder(tf.bool)
    global_step_g = tf.get_variable('global_step_g',
                                    [],
                                    tf.int32,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)

    global_step_d = tf.get_variable('global_step_d',
                                    [],
                                    tf.int32,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(input_parse)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    # iterator_d = dataset.make_initializable_iterator()
    images, images_with_hole, masks, x_locs, y_locs, hole_heights, hole_widths = iterator.get_next()

    syn_images = completion_network(images_with_hole, batch_size)
    # completed_images = (1 - masks) * images + masks * syn_images
    completed_images = tf.multiply(1 - masks, images) + tf.multiply(masks, syn_images)

    local_dis_inputs_fake = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0],
                                                                                    args[1],
                                                                                    args[2],
                                                                                    gt_height,
                                                                                    gt_width),
                                      elems=(completed_images, y_locs, x_locs),
                                      dtype=tf.float32)
    local_dis_inputs_real = tf.map_fn(fn=lambda args: tf.image.crop_to_bounding_box(args[0],
                                                                                    args[1],
                                                                                    args[2],
                                                                                    gt_height,
                                                                                    gt_width),
                                      elems=(images, y_locs, x_locs),
                                      dtype=tf.float32)

    # loss function
    # hole_heights = tf.convert_to_tensor(hole_heights)
    # hole_widths = tf.convert_to_tensor(hole_widths)
    sizes = 3 * tf.multiply(hole_heights, hole_widths)
    temp = tf.abs(completed_images - images)
    loss_recon = tf.reduce_mean(tf.div(tf.reduce_sum(temp, axis=[1, 2, 3]), sizes))
    # loss_recon = tf.reduce_mean([tf.div(temp[i], sizes[i]) for i in range(batch_size)])
    # loss_recon = tf.reduce_mean(tf.abs(completed_images - images))
    # loss_recon = tf.reduce_mean(tf.abs(masks * (syn_images - images)))

    # loss function only for training generator
    loss_only_g = loss_recon + weight_decay_rate * tf.reduce_mean(tf.get_collection('weight_decay_gen'))

    global_dis_outputs_real = global_discriminator(images)
    global_dis_outputs_fake = global_discriminator(completed_images, reuse=True)
    global_dis_outputs_all = tf.concat([global_dis_outputs_real, global_dis_outputs_fake], axis=0)

    local_dis_outputs_real = markovian_discriminator(local_dis_inputs_real)
    local_dis_outputs_fake = markovian_discriminator(local_dis_inputs_fake, reuse=True)
    local_dis_outputs_all = tf.concat([local_dis_outputs_real, local_dis_outputs_fake], axis=0)

    labels_global_dis = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)
    labels_local_dis = tf.concat([tf.ones_like(local_dis_outputs_real),
                                  tf.zeros_like(local_dis_outputs_fake)], axis=0)

    loss_global_dis = 2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=global_dis_outputs_all,
        labels=labels_global_dis
    ))

    loss_local_dis = 2 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=local_dis_outputs_all,
        labels=labels_local_dis
    ))

    loss_global_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=global_dis_outputs_fake,
        labels=tf.ones([batch_size])
    ))
    loss_local_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=local_dis_outputs_fake,
        labels=tf.ones_like(local_dis_outputs_fake)
    ))

    loss_g = alpha_rec * loss_recon + alpha_global * loss_global_g + alpha_local * loss_local_g
    loss_d = loss_global_dis + loss_local_dis

    var_g = tf.get_collection('gen_params_conv')  # + tf.get_collection('gen_params_bn')
    var_d = tf.get_collection('global_dis_params_conv') + tf.get_collection('local_dis_params_conv')
    # tf.get_collection('global_dis_params_bn') +\
    # tf.get_collection('local_dis_params_conv') +\
    # tf.get_collection('local_dis_params_bn')

    lr_g = tf.train.exponential_decay(learning_rate=init_lr_g,
                                      global_step=global_step_g,
                                      decay_steps=lr_decay_steps,
                                      decay_rate=0.98)

    lr_d = tf.train.exponential_decay(learning_rate=init_lr_d,
                                      global_step=global_step_d,
                                      decay_steps=lr_decay_steps,
                                      decay_rate=0.97)

    opt_g = tf.train.AdamOptimizer(learning_rate=lr_g, beta1=0.5)
    opt_d = tf.train.AdamOptimizer(learning_rate=lr_d, beta1=0.5)

    # grads and vars
    grads_vars_only_g = opt_g.compute_gradients(loss_only_g, var_g)
    train_only_g = opt_g.apply_gradients(grads_vars_only_g, global_step_g)

    grads_vars_g = opt_g.compute_gradients(loss_g, var_g)
    train_g = opt_g.apply_gradients(grads_vars_g, global_step_g)

    grads_vars_d = opt_d.compute_gradients(loss_d, var_d)
    train_d = opt_d.apply_gradients(grads_vars_d, global_step_d)

    # view grads and vars
    view_only_g_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_only_g])
    view_only_g_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_only_g])

    view_g_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_g])
    view_g_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_g])

    view_d_grads = tf.reduce_mean([tf.reduce_mean(gv[0]) if gv[0] is not None else 0. for gv in grads_vars_d])
    view_d_weights = tf.reduce_mean([tf.reduce_mean(gv[1]) for gv in grads_vars_d])

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(decay=0.999)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op_only_g = tf.group(train_only_g, variable_averages_op)
    train_op_g = tf.group(train_g, variable_averages_op)
    train_op_d = tf.group(train_d, variable_averages_op)

    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # if isFirstTimeTrain:
    #     old_var_G = []
    #     graph1 = tf.Graph()
    #     with graph1.as_default():
    #         with tf.Session(graph=graph1) as sess1:
    #             saver1 = tf.train.import_meta_graph(os.path.join(g_model_path, 'models_without_adv_l1.meta'))
    #             saver1.restore(sess1, os.path.join(g_model_path, 'models_without_adv_l1'))
    #             old_var_G = tf.get_collection('gen_params_conv') + tf.get_collection('gen_params_bn')
    #             old_var_G = sess1.run(old_var_G)

    with tf.Session() as sess:
        # load trainset
        train_path = pd.read_pickle(compress_path)
        train_path.index = range(len(train_path))
        train_path = train_path.ix[np.random.permutation(len(train_path))]
        train_path = train_path[:]['image_path'].values.tolist()
        num_batch = int(len(train_path) / batch_size)

        sess.run(iterator.initializer, feed_dict={filenames: train_path})
        sess.run(tf.global_variables_initializer())

        if isFirstTimeTrain:
            # sess.run(tf.global_variables_initializer())
            # updates = []
            # for i, item in enumerate(old_var_G):
            #     updates.append(tf.assign(var_g[i], item))
            # sess.run(updates)

            iters = 0
            with open(os.path.join(g_model_path, 'iter.pickle'), 'wb') as f:
                pickle.dump(iters, f, protocol=2)
            saver.save(sess, os.path.join(g_model_path, 'models_without_adv_l1'))
        else:
            # sess.run(tf.global_variables_initializer())
            # saver.restore(sess, os.path.join(model_path, 'models_global_local_l1'))
            with open(os.path.join(g_model_path, 'iter.pickle'), 'rb') as f:
                iters = pickle.load(f)
            saver.restore(sess, os.path.join(g_model_path, 'models_without_adv_l1'))

            if iters == iters_c:
                if isFirstTimeTrain_Joint:
                    iters += 1
                else:
                    with open(os.path.join(model_path, 'iter.pickle'), 'rb') as f:
                        iters = pickle.load(f)
                    # iters += 1
                    saver.restore(sess, os.path.join(model_path, 'models_global_local_l1'))

        while iters <= iters_total:
            if iters <= iters_c:
                _, loss_view_g, lr_view_g, gs = \
                    sess.run([train_op_only_g, loss_only_g, lr_g, global_step_g])
                #  feed_dict={is_training: True})
                print('Epoch: {}, Iter: {},loss_g: {},lr_g: {}'.format(
                    int(iters / num_batch) + 1,
                    gs,  # iters,
                    loss_view_g,
                    lr_view_g))

                if (iters % 200 == 0) or (iters == iters_c):
                    with open(os.path.join(g_model_path, 'iter.pickle'), 'wb') as f:
                        pickle.dump(iters, f, protocol=2)
                    saver.save(sess, os.path.join(g_model_path, 'models_without_adv_l1'))

                    g_vars_mean, g_grads_mean = sess.run([view_only_g_weights,
                                                          view_only_g_grads])
                    #  feed_dict={is_training: True})
                    # summary_writer.add_summary(summary_str, iters)
                    print('Epoch: {}, Iter: {}, g_weights_mean: {}, g_grads_mean: {}'.format(
                        int(iters / num_batch) + 1,
                        gs,
                        g_vars_mean,
                        g_grads_mean))

            else:
                _, _, _, loss_view_g, loss_view_d, lr_view_g, lr_view_d, gs = \
                    sess.run([train_op_g, train_op_g, train_op_d, loss_g, loss_d, lr_g, lr_d, global_step_d])
                #  feed_dict={is_training: True})

                print('Epoch: {}, Iter: {}, loss_d: {},loss_g: {}, lr_d: {}, lr_g: {}'.format(
                    int(iters / num_batch) + 1,
                    gs,  # iters,
                    loss_view_d,
                    loss_view_g,
                    lr_view_d,
                    lr_view_g))

                if (iters % 200 == 0) or (iters == iters_total):
                    with open(os.path.join(model_path, 'iter.pickle'), 'wb') as f:
                        pickle.dump(iters, f, protocol=2)
                    saver.save(sess, os.path.join(model_path, 'models_global_local_l1'))

                    g_vars_mean, g_grads_mean, d_vars_mean, d_grads_mean = sess.run([view_g_weights,
                                                                                     view_g_grads,
                                                                                     view_d_weights,
                                                                                     view_d_grads])
                    # feed_dict={is_training: True})
                    # summary_writer.add_summary(summary_str, iters)
                    print('Epoch: {}, Iter: {}, g_weights_mean: {}, g_grads_mean: {}'.format(
                        int(iters / num_batch) + 1,
                        gs,
                        g_vars_mean,
                        g_grads_mean))
                    print('-------------------d_weights_mean: {}, d_grads_mean: {}'.format(d_vars_mean,
                                                                                           d_grads_mean))
            iters += 1


if __name__ == '__main__':
    train()
    print('done.')
    # batch_size = 100
    # imgs = tf.placeholder(tf.float32, [batch_size, 96, 96, 3])
    # train_flag = tf.placeholder(tf.bool)

    # result = markovian_discriminator(imgs, train_flag)
    # print(result.get_shape())
