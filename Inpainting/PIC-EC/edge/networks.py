from __future__ import print_function
import os
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss

from metrics import edge_accuracy


class EdgeModel():
    """Construct edge model."""

    def __init__(self, config=None):
        print('Construct the edge model.')
        self.cfg = config
        self.init_type = self.cfg['INIT_TYPE']

    def edge_generator(self, x, reuse=None):
        with tf.variable_scope('edge_generator', reuse=reuse):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1,
                     pad_type='zero', init_type=self.init_type, name='conv3')
            x = instance_norm(x, name='in3')
            x = tf.nn.relu(x)

            # resnet block
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block1')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block2')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block3')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block4')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block5')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block6')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block7')
            x = resnet_block(x, out_channels=256, dilation=2, init_type=self.init_type, name='resnet_block8')

            # decoder
            shape1 = tf.shape(x)
            x = tf.image.resize_nearest_neighbor(x, size=(shape1[1] * 2, shape1[2] * 2))
            x = conv(x, channels=128, kernel=3, stride=1, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv4')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            shape2 = tf.shape(x)
            x = tf.image.resize_nearest_neighbor(x, size=(shape2[1] * 2, shape2[2] * 2))
            x = conv(x, channels=64, kernel=3, stride=1, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv5')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=1, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv6')
            edge = tf.nn.sigmoid(x)
            edge = tf.cast(edge > 0.25, tf.float32)

            return edge, x

    def edge_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('edge_discriminator', reuse=reuse):
            conv1 = conv(x, channels=64, kernel=4, stride=2, pad=1, pad_type='zero',
                         use_bias=False, init_type=self.init_type, name='conv1')
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = conv(conv1, channels=128, kernel=4, stride=2, pad=1, pad_type='zero',
                         use_bias=False, init_type=self.init_type, name='conv2')
            conv2 = tf.nn.leaky_relu(conv2)

            conv3 = conv(conv2, channels=256, kernel=4, stride=2, pad=1, pad_type='zero',
                         use_bias=False, init_type=self.init_type, name='conv3')
            conv3 = tf.nn.leaky_relu(conv3)

            conv4 = conv(conv3, channels=512, kernel=4, stride=1, pad=1, pad_type='zero',
                         use_bias=False, init_type=self.init_type, name='conv4')
            conv4 = tf.nn.leaky_relu(conv4)

            conv5 = conv(conv4, channels=1, kernel=4, stride=1, pad=1, pad_type='zero',
                         use_bias=False, init_type=self.init_type, name='conv5')

            outputs = conv5
            if use_sigmoid:
                outputs = tf.nn.sigmoid(conv5)

            return outputs, [conv1, conv2, conv3, conv4, conv5]

    def build_model(self, img_grays, edges, masks):
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: [grayscale(1) + edge(1)]
        edges_masked = edges * (1 - masks)
        grays_masked = img_grays * (1 - masks) + masks
        # edges_masked = edges
        # grays_masked = img_grays
        inputs = tf.concat([grays_masked, edges_masked, masks * tf.ones_like(img_grays)], axis=3)
        outputs, logits = self.edge_generator(inputs)
        outputs_merged = outputs * masks + edges * (1 - masks)

        # metrics
        precision, recall = edge_accuracy(edges * masks, outputs_merged * masks, self.cfg['EDGE_THRESHOLD'])

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # get the global steps
        gen_global_step = tf.get_variable('gen_global_step',
                                          [],
                                          tf.int32,
                                          initializer=tf.zeros_initializer(),
                                          trainable=False)
        dis_global_step = tf.get_variable('dis_global_step',
                                          [],
                                          tf.int32,
                                          initializer=tf.zeros_initializer(),
                                          trainable=False)

        gen_loss = 0.0
        dis_loss = 0.0

        # discriminator loss
        dis_input_real = tf.concat([img_grays, edges], axis=3)
        # dis_input_fake = tf.concat([img, tf.stop_gradient(outputs_merged)], axis=3)
        dis_input_fake = tf.concat([img_grays, tf.stop_gradient(outputs_merged)], axis=3)
        dis_real, dis_real_feat = self.edge_discriminator(dis_input_real, use_sigmoid=use_sigmoid)
        dis_fake, dis_fake_feat = self.edge_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True,
                                         gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False,
                                         gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_fake_loss + dis_real_loss) / 2.0

        # generator adversarial loss
        gen_input_fake = tf.concat([img_grays, outputs_merged], axis=3)
        # gen_input_fake = tf.concat([imgs, outputs_merged], axis=3)
        gen_fake, gen_fake_feat = self.edge_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True,
                                        gan_type=self.cfg['GAN_LOSS'], is_disc=False)

        # generator feature matching loss
        gen_fm_loss = 0.0
        for (real_feat, fake_feat) in zip(dis_real_feat, gen_fake_feat):
            gen_fm_loss += tf.losses.absolute_difference(tf.stop_gradient(real_feat), fake_feat)

        # generator cross entropy loss
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=edges, logits=logits)
        hole_ce = tf.reduce_mean(tf.reduce_mean(ce * masks, axis=(1, 2, 3)) / tf.reduce_mean(masks, axis=(1, 2, 3)))
        unhole_ce = tf.reduce_mean(tf.reduce_mean(ce * (1 - masks), axis=(1, 2, 3)) /
                                   tf.reduce_mean(1 - masks, axis=(1, 2, 3)))
        l2 = tf.square(edges - outputs)
        hole_l2 = tf.reduce_mean(tf.reduce_mean(l2 * masks, axis=(1, 2, 3)) / tf.reduce_mean(masks, axis=(1, 2, 3)))
        unhole_l2 = tf.reduce_mean(tf.reduce_mean(l2 * (1 - masks), axis=(1, 2, 3)) /
                                   tf.reduce_mean(1 - masks, axis=(1, 2, 3)))

        gen_ce_loss = 5 * hole_l2 * hole_ce + unhole_l2 * unhole_ce

        # all loss
        gen_loss = gen_gan_loss * self.cfg['ADV_LOSS_WEIGHT'] +\
            gen_fm_loss * self.cfg['FM_LOSS_WEIGHT'] +\
            gen_ce_loss * self.cfg['CE_LOSS_WEIGHT']

        # get model variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'edge_generator')
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'edge_discriminator')

        # get the optimizer for training
        gen_opt = tf.train.AdamOptimizer(self.cfg['LR'],
                                         beta1=self.cfg['BETA1'],
                                         beta2=self.cfg['BETA2'])
        dis_opt = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                         beta1=self.cfg['BETA1'],
                                         beta2=self.cfg['BETA2'])

        # optimize the model
        gen_train = gen_opt.minimize(gen_loss,
                                     global_step=gen_global_step,
                                     var_list=gen_vars)
        dis_train = dis_opt.minimize(dis_loss,
                                     global_step=dis_global_step,
                                     var_list=dis_vars)

        dis_train_ops = []
        for i in range(3):
            dis_train_ops.append(dis_train)
        dis_train = tf.group(*dis_train_ops)

        # create logs
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_fm_loss, gen_ce_loss]

        # add summary for monitor
        tf.summary.scalar('train_dis_loss', dis_loss)
        tf.summary.scalar('train_gen_loss', gen_loss)
        tf.summary.scalar('train_gen_gan_loss', gen_gan_loss)
        tf.summary.scalar('train_gen_fm_loss', gen_fm_loss)
        tf.summary.scalar('train_gen_ce_loss', gen_ce_loss)
        tf.summary.scalar('train_precision', precision)
        tf.summary.scalar('train_recall', recall)

        return gen_train, dis_train, logs

    def eval_model(self, img_grays, edges, masks):
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        edges_masked = edges * (1 - masks)
        grays_masked = img_grays * (1 - masks) + masks
        inputs = tf.concat([grays_masked, edges_masked, masks * tf.ones_like(img_grays)], axis=3)
        outputs, _ = self.edge_generator(inputs, reuse=True)
        outputs_merged = outputs * masks + edges * (1 - masks)
        # outputs_merged = tf.clip_by_value(outputs_merged, 0, 1)

        # metrics
        precision, recall = edge_accuracy(edges * masks, outputs_merged * masks, self.cfg['EDGE_THRESHOLD'])

        tf.summary.scalar('val_precision', precision)
        tf.summary.scalar('val_recall', recall)

        visual_img = [grays_masked, edges, edges_masked, outputs_merged]
        visual_img = tf.concat(visual_img, axis=2)
        tf.summary.image('gray_edge_merged', visual_img, 4)

        val_logs = [precision, recall]
        return val_logs

    def test_model(self, img_grays, edges, masks):
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        edges_masked = edges * (1 - masks)
        grays_masked = img_grays * (1 - masks) + masks
        inputs = tf.concat([grays_masked, edges_masked, masks * tf.ones_like(img_grays)], axis=3)
        outputs, _ = self.edge_generator(inputs)
        outputs_merged = outputs * masks + edges * (1 - masks)

        return outputs_merged

    def sample(self, img_grays, edges, masks):
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        edges_masked = edges * (1 - masks)
        grays_masked = img_grays * (1 - masks) + masks
        inputs = tf.concat([grays_masked, edges_masked, masks * tf.ones_like(img_grays)], axis=3)
        outputs, _ = self.edge_generator(inputs)
        outputs_merged = outputs * masks + edges * (1 - masks)

        return outputs_merged

    def save(self, sess, saver, path, model_name):
        print('\nsaving the model...\n')
        saver.save(sess, os.path.join(path, model_name))

    def load(self, sess, saver, path, model_name):
        print('\nloading the model...\n')
        saver.restore(sess, os.path.join(path, model_name))
