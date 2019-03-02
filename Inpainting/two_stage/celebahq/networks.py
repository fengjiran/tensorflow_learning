from __future__ import print_function
import os
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss
from loss import perceptual_loss
from loss import style_loss
from loss import Vgg19

from utils import images_summary


class InpaintingModel():
    """Construct model."""

    def __init__(self, config=None):
        print('Construct the inpainting model.')
        self.cfg = config
        self.init_type = self.cfg['INIT_TYPE']
        self.vgg = Vgg19()

        # global step for training
        # self.gen_global_step = tf.get_variable('gen_global_step',
        #                                        [],
        #                                        tf.int32,
        #                                        initializer=tf.zeros_initializer(),
        #                                        trainable=False)
        # self.dis_global_step = tf.get_variable('dis_global_step',
        #                                        [],
        #                                        tf.int32,
        #                                        initializer=tf.zeros_initializer(),
        #                                        trainable=False)

        # self.coarse_gen_vars = None
        # self.coarse_dis_vars = None
        # self.refine_gen_vars = None
        # self.refine_dis_vars = None

    def coarse_generator(self, x, reuse=None):
        with tf.variable_scope('coarse_generator', reuse=reuse):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1,
                     pad_type='zero', init_type=self.init_type, name='conv2')
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
            x = deconv(x, channels=128, kernel=4, stride=2, init_type=self.init_type, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            x = deconv(x, channels=64, kernel=4, stride=2, init_type=self.init_type, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv4')
            x = tf.nn.tanh(x)

            return x  # [-1, 1]

    def coarse_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('coarse_discriminator', reuse=reuse):
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

    """
    def build_coarse_model(self, images, masks):
        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks

        inputs = tf.concat([images_masked, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        outputs = self.coarse_generator(inputs)
        outputs_merged = outputs * masks + images * (1.0 - masks)

        dis_loss = 0.0
        gen_loss = 0.0

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # discriminator loss
        dis_input_real = images
        dis_input_fake = tf.stop_gradient(outputs)
        dis_real, _ = self.coarse_discriminator(dis_input_real, use_sigmoid=use_sigmoid)
        dis_fake, _ = self.coarse_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2.0

        # generator adversartial loss
        gen_input_fake = outputs
        gen_fake, _ = self.coarse_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(
            gen_fake, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=False) * self.cfg['ADV_LOSS_WEIGHT']
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = tf.losses.absolute_difference(
            images, outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = perceptual_loss(outputs, images, self.vgg) * self.cfg['CONTENT_LOSS_WEIGHT']
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = style_loss(outputs * masks, images * masks, self.vgg) * self.cfg['STYLE_LOSS_WEIGHT']
        gen_loss += gen_style_loss

        # get coarse model variables
        coarse_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_generator')
        coarse_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_discriminator')

        # get the optimizer for training
        coarse_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])
        coarse_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])

        # global step for training
        coarse_gen_global_step = tf.get_variable('coarse_gen_global_step',
                                                 [],
                                                 tf.int32,
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=False)
        coarse_dis_global_step = tf.get_variable('coarse_dis_global_step',
                                                 [],
                                                 tf.int32,
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=False)

        # optimize the model
        coarse_gen_train = coarse_gen_optimizer.minimize(gen_loss,
                                                         global_step=coarse_gen_global_step,
                                                         var_list=coarse_gen_vars)
        coarse_dis_train = coarse_dis_optimizer.minimize(dis_loss,
                                                         global_step=coarse_dis_global_step,
                                                         var_list=coarse_dis_vars)

        # create logs
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_style_loss, gen_content_loss]

        # add summary for monitor
        tf.summary.scalar('coarse_dis_loss', dis_loss)
        tf.summary.scalar('coarse_gen_loss', gen_loss)
        tf.summary.scalar('coarse_gen_gan_loss', gen_gan_loss)
        tf.summary.scalar('coarse_gen_l1_loss', gen_l1_loss)
        tf.summary.scalar('coarse_gen_style_loss', gen_style_loss)
        tf.summary.scalar('coarse_gen_content_loss', gen_content_loss)

        visual_img = [images, images_masked, outputs_merged]
        visual_img = tf.concat(visual_img, axis=2)
        images_summary(visual_img, 'gt_masked_inpainted', 4)

        return outputs, outputs_merged, coarse_gen_train, coarse_dis_train, logs
    """

    def refine_generator(self, x, reuse=None):
        with tf.variable_scope('refine_generator', reuse=reuse):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1,
                     pad_type='zero', init_type=self.init_type, name='conv2')
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
            x = deconv(x, channels=128, kernel=4, stride=2, init_type=self.init_type, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            x = deconv(x, channels=64, kernel=4, stride=2, init_type=self.init_type, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv4')
            x = tf.nn.tanh(x)

            return x  # [-1, 1]

    def refine_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('refine_discriminator', reuse=reuse):
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

    """
    def build_refine_model(self, images, masks):
        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks
        coarse_inputs = tf.concat([images_masked, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        coarse_outputs = self.coarse_generator(coarse_inputs, reuse=True)
        coarse_outputs = tf.stop_gradient(coarse_outputs)
        coarse_outputs_merged = coarse_outputs * masks + images * (1.0 - masks)

        refine_inputs = tf.concat([coarse_outputs_merged, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        refine_outputs = self.refine_generator(refine_inputs)
        refine_outputs_merged = refine_outputs * masks + images * (1.0 - masks)

        dis_loss = 0.0
        gen_loss = 0.0

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # discriminator loss
        dis_input_real = images
        dis_input_fake = tf.stop_gradient(refine_outputs)
        dis_real, _ = self.refine_discriminator(dis_input_real, use_sigmoid=use_sigmoid)
        dis_fake, _ = self.refine_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2.0

        # generator adversartial loss
        gen_input_fake = refine_outputs
        gen_fake, _ = self.refine_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=False)
        gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = tf.losses.absolute_difference(
            images, refine_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = perceptual_loss(refine_outputs, images, self.vgg) * self.cfg['CONTENT_LOSS_WEIGHT']
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = style_loss(refine_outputs * masks, images * masks, self.vgg) * self.cfg['STYLE_LOSS_WEIGHT']
        gen_loss += gen_style_loss

        # get the refine model variables
        refine_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_generator')
        refine_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_discriminator')

        # get the refine optimizers
        refine_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])
        refine_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])

        # get the global steps
        refine_gen_global_step = tf.get_variable('refine_gen_global_step',
                                                 [],
                                                 tf.int32,
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=False)
        refine_dis_global_step = tf.get_variable('refine_dis_global_step',
                                                 [],
                                                 tf.int32,
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=False)

        # optimize the refine models
        refine_gen_train = refine_gen_optimizer.minimize(gen_loss,
                                                         global_step=refine_gen_global_step,
                                                         var_list=refine_gen_vars)
        refine_dis_train = refine_dis_optimizer.minimize(dis_loss,
                                                         global_step=refine_dis_global_step,
                                                         var_list=refine_dis_vars)

        # create logs
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_style_loss, gen_content_loss]

        # add summary for monitor
        # tf.summary.scalar('refine_dis_loss', dis_loss)
        # tf.summary.scalar('refine_gen_loss', gen_loss)
        # tf.summary.scalar('refine_gen_gan_loss', gen_gan_loss)
        # tf.summary.scalar('refine_gen_l1_loss', gen_l1_loss)
        # tf.summary.scalar('refine_gen_style_loss', gen_style_loss)
        # tf.summary.scalar('refine_gen_content_loss', gen_content_loss)

        # visual_img = [images, images_masked, coarse_outputs_merged, refine_outputs_merged]
        # visual_img = tf.concat(visual_img, axis=2)
        # images_summary(visual_img, 'gt_masked_coarse_refine', 4)

        return refine_outputs, refine_outputs_merged, refine_gen_train, refine_dis_train, logs
    """

    """
    def build_joint_model(self, images, masks):
        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks
        coarse_inputs = tf.concat([images_masked, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        coarse_outputs = self.coarse_generator(coarse_inputs, reuse=True)
        coarse_outputs_merged = coarse_outputs * masks + images * (1.0 - masks)

        refine_inputs = tf.concat([coarse_outputs_merged, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        refine_outputs = self.refine_generator(refine_inputs, reuse=True)
        refine_outputs_merged = refine_outputs * masks + images * (1.0 - masks)

        dis_loss = 0.0
        gen_loss = 0.0

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # discriminator loss
        dis_input_real = images
        dis_input_fake = tf.stop_gradient(refine_outputs)
        dis_real, _ = self.refine_discriminator(dis_input_real, reuse=True, use_sigmoid=use_sigmoid)
        dis_fake, _ = self.refine_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2.0

        # generator adversartial loss
        gen_input_fake = refine_outputs
        gen_fake, _ = self.refine_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=False)
        gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = tf.losses.absolute_difference(
            images, refine_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = perceptual_loss(refine_outputs, images, self.vgg) * self.cfg['CONTENT_LOSS_WEIGHT']
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = style_loss(refine_outputs * masks, images * masks, self.vgg) * self.cfg['STYLE_LOSS_WEIGHT']
        gen_loss += gen_style_loss

        # get the joint model variables
        coarse_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_generator')
        # coarse_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_discriminator')
        refine_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_generator')
        refine_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_discriminator')

        joint_gen_vars = coarse_gen_vars + refine_gen_vars
        joint_dis_vars = refine_dis_vars

        joint_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                     beta1=self.cfg['BETA1'],
                                                     beta2=self.cfg['BETA2'])
        joint_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                     beta1=self.cfg['BETA1'],
                                                     beta2=self.cfg['BETA2'])

        # get the global steps
        joint_gen_global_step = tf.get_variable('joint_gen_global_step',
                                                [],
                                                tf.int32,
                                                initializer=tf.zeros_initializer(),
                                                trainable=False)
        joint_dis_global_step = tf.get_variable('joint_dis_global_step',
                                                [],
                                                tf.int32,
                                                initializer=tf.zeros_initializer(),
                                                trainable=False)

        joint_gen_train = joint_gen_optimizer.minimize(gen_loss,
                                                       global_step=joint_gen_global_step,
                                                       var_list=joint_gen_vars)
        joint_dis_train = joint_dis_optimizer.minimize(dis_loss,
                                                       global_step=joint_dis_global_step,
                                                       var_list=joint_dis_vars)

        # create logs
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_style_loss, gen_content_loss]

        # add summary for monitor
        # tf.summary.scalar('joint_dis_loss', dis_loss)
        # tf.summary.scalar('joint_gen_loss', gen_loss)
        # tf.summary.scalar('joint_gen_gan_loss', gen_gan_loss)
        # tf.summary.scalar('joint_gen_l1_loss', gen_l1_loss)
        # tf.summary.scalar('joint_gen_style_loss', gen_style_loss)
        # tf.summary.scalar('joint_gen_content_loss', gen_content_loss)

        # # add summaries for image visual
        # visual_img = [images, images_masked, coarse_outputs_merged, refine_outputs_merged]
        # visual_img = tf.concat(visual_img, axis=2)
        # images_summary(visual_img, 'gt_masked_coarse_refine', 4)

        return refine_outputs, refine_outputs_merged, joint_gen_train, joint_dis_train, logs
    """

    # def test(self, images, masks):
    #     images_masked = images * (1.0 - masks) + masks
    #     coarse_inputs = tf.concat([images_masked, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
    #     coarse_outputs = self.coarse_generator(coarse_inputs)
    #     coarse_outputs_merged = coarse_outputs * masks + images * (1.0 - masks)

    #     refine_inputs = tf.concat([coarse_outputs_merged, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
    #     refine_outputs = self.refine_generator(refine_inputs)
    #     refine_outputs_merged = refine_outputs * masks + images * (1.0 - masks)

    #     if self.cfg['GAN_LOSS'] == 'lsgan':
    #         use_sigmoid = True
    #     else:
    #         use_sigmoid = False

    #     # get the global steps
    #     gen_global_step = tf.get_variable('gen_global_step',
    #                                       [],
    #                                       tf.int32,
    #                                       initializer=tf.zeros_initializer(),
    #                                       trainable=False)
    #     dis_global_step = tf.get_variable('dis_global_step',
    #                                       [],
    #                                       tf.int32,
    #                                       initializer=tf.zeros_initializer(),
    #                                       trainable=False)

    #     # --------------------- Build coarse loss function -----------------------------
    #     coarse_gen_loss = 0.0
    #     coarse_dis_loss = 0.0

    #     # discriminator loss
    #     coarse_dis_input_real = images
    #     coarse_dis_input_fake = tf.stop_gradient(coarse_outputs)
    #     coarse_dis_real, _ = self.coarse_discriminator(coarse_dis_input_real, use_sigmoid=use_sigmoid)
    #     coarse_dis_fake, _ = self.coarse_discriminator(coarse_dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
    #     coarse_dis_real_loss = adversarial_loss(coarse_dis_real, is_real=True,
    #                                             gan_type=self.cfg['GAN_LOSS'], is_disc=True)
    #     coarse_dis_fake_loss = adversarial_loss(coarse_dis_fake, is_real=False,
    #                                             gan_type=self.cfg['GAN_LOSS'], is_disc=True)
    #     coarse_dis_loss += (coarse_dis_real_loss + coarse_dis_fake_loss) / 2.0

    #     # generator adversartial loss
    #     coarse_gen_input_fake = coarse_outputs
    #     coarse_gen_fake, _ = self.coarse_discriminator(coarse_gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
    #     coarse_gen_gan_loss = adversarial_loss(coarse_gen_fake, is_real=True,
    #                                            gan_type=self.cfg['GAN_LOSS'], is_disc=False)
    #     coarse_gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
    #     coarse_gen_loss += coarse_gen_gan_loss

    #     # generator l1 loss
    #     coarse_gen_l1_loss = tf.losses.absolute_difference(
    #         images, coarse_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
    #     coarse_gen_loss += coarse_gen_l1_loss

    #     # generator perceptual loss
    #     x = self.vgg.forward(coarse_outputs)
    #     y = self.vgg.forward(images, reuse=True)
    #     coarse_gen_content_loss = perceptual_loss(x, y) * self.cfg['CONTENT_LOSS_WEIGHT']
    #     coarse_gen_loss += coarse_gen_content_loss

    #     # generator style loss
    #     m = self.vgg.forward(coarse_outputs * masks, reuse=True)
    #     n = self.vgg.forward(images * masks, reuse=True)
    #     coarse_gen_style_loss = style_loss(m, n) * self.cfg['STYLE_LOSS_WEIGHT']
    #     coarse_gen_loss += coarse_gen_style_loss

    #     temp = [coarse_outputs_merged, refine_outputs_merged]

    #     return temp

    def build_model(self, images, masks):
        # generator input: [rgb(3)+mask(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks
        coarse_inputs = tf.concat([images_masked, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        coarse_outputs = self.coarse_generator(coarse_inputs)
        coarse_outputs_merged = coarse_outputs * masks + images * (1.0 - masks)

        refine_inputs = tf.concat([coarse_outputs_merged, masks * (tf.ones_like(images)[:, :, :, 0:1])], axis=3)
        refine_outputs = self.refine_generator(refine_inputs)
        refine_outputs_merged = refine_outputs * masks + images * (1.0 - masks)

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

        # --------------------- Build coarse loss function -----------------------------
        # coarse_gen_loss = 0.0
        # coarse_dis_loss = 0.0

        # discriminator loss
        coarse_dis_input_real = images
        coarse_dis_input_fake = tf.stop_gradient(coarse_outputs)
        coarse_dis_real, _ = self.coarse_discriminator(coarse_dis_input_real, use_sigmoid=use_sigmoid)
        coarse_dis_fake, _ = self.coarse_discriminator(coarse_dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        coarse_dis_real_loss = adversarial_loss(coarse_dis_real, is_real=True,
                                                gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        coarse_dis_fake_loss = adversarial_loss(coarse_dis_fake, is_real=False,
                                                gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        coarse_dis_loss = (coarse_dis_real_loss + coarse_dis_fake_loss) / 2.0

        # generator adversartial loss
        coarse_gen_input_fake = coarse_outputs
        coarse_gen_fake, _ = self.coarse_discriminator(coarse_gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        coarse_gen_gan_loss = adversarial_loss(coarse_gen_fake, is_real=True,
                                               gan_type=self.cfg['GAN_LOSS'], is_disc=False)
        coarse_gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
        coarse_gen_loss = coarse_gen_gan_loss

        # generator l1 loss
        coarse_gen_l1_loss = tf.losses.absolute_difference(
            images, coarse_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        coarse_gen_loss += coarse_gen_l1_loss

        # generator perceptual loss
        coarse_content_x = self.vgg.forward(coarse_outputs)
        coarse_content_y = self.vgg.forward(images, reuse=True)
        coarse_gen_content_loss = perceptual_loss(coarse_content_x, coarse_content_y) * self.cfg['CONTENT_LOSS_WEIGHT']
        coarse_gen_loss += coarse_gen_content_loss

        # generator style loss
        coarse_style_x = self.vgg.forward(coarse_outputs * masks, reuse=True)
        coarse_style_y = self.vgg.forward(images * masks, reuse=True)
        coarse_gen_style_loss = style_loss(coarse_style_x, coarse_style_y) * self.cfg['STYLE_LOSS_WEIGHT']
        coarse_gen_loss += coarse_gen_style_loss

        # get coarse model variables
        coarse_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_generator')
        coarse_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_discriminator')

        # get the optimizer for training
        coarse_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])
        coarse_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])

        # optimize the model
        coarse_gen_train = coarse_gen_optimizer.minimize(coarse_gen_loss,
                                                         global_step=gen_global_step,
                                                         var_list=coarse_gen_vars)
        coarse_dis_train = coarse_dis_optimizer.minimize(coarse_dis_loss,
                                                         global_step=dis_global_step,
                                                         var_list=coarse_dis_vars)

        coarse_dis_train_ops = []
        for i in range(5):
            coarse_dis_train_ops.append(coarse_dis_train)
        coarse_dis_train = tf.group(*coarse_dis_train_ops)

        # create logs
        coarse_logs = [coarse_dis_loss, coarse_gen_loss, coarse_gen_gan_loss,
                       coarse_gen_l1_loss, coarse_gen_style_loss, coarse_gen_content_loss]

        # add summary for monitor
        tf.summary.scalar('coarse_dis_loss', coarse_dis_loss)
        tf.summary.scalar('coarse_gen_loss', coarse_gen_loss)
        tf.summary.scalar('coarse_gen_gan_loss', coarse_gen_gan_loss)
        tf.summary.scalar('coarse_gen_l1_loss', coarse_gen_l1_loss)
        tf.summary.scalar('coarse_gen_style_loss', coarse_gen_style_loss)
        tf.summary.scalar('coarse_gen_content_loss', coarse_gen_content_loss)

        coarse_visual_img = [images, images_masked, coarse_outputs_merged]
        coarse_visual_img = tf.concat(coarse_visual_img, axis=2)
        images_summary(coarse_visual_img, 'coarse_gt_masked_inpainted', 4)

        coarse_returned = [coarse_outputs, coarse_outputs_merged, coarse_gen_train, coarse_dis_train, coarse_logs]

        # --------------------- Build refine loss function -----------------------------
        refine_gen_loss = 0.0
        refine_dis_loss = 0.0

        # discriminator loss
        refine_dis_input_real = images
        refine_dis_input_fake = tf.stop_gradient(refine_outputs)
        refine_dis_real, _ = self.refine_discriminator(refine_dis_input_real, use_sigmoid=use_sigmoid)
        refine_dis_fake, _ = self.refine_discriminator(refine_dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        refine_dis_real_loss = adversarial_loss(refine_dis_real, is_real=True,
                                                gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        refine_dis_fake_loss = adversarial_loss(refine_dis_fake, is_real=False,
                                                gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        refine_dis_loss += (refine_dis_real_loss + refine_dis_fake_loss) / 2.0

        # generator adversartial loss
        refine_gen_input_fake = refine_outputs
        refine_gen_fake, _ = self.refine_discriminator(refine_gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        refine_gen_gan_loss = adversarial_loss(refine_gen_fake, is_real=True,
                                               gan_type=self.cfg['GAN_LOSS'], is_disc=False)
        refine_gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
        refine_gen_loss += refine_gen_gan_loss

        # generator l1 loss
        refine_gen_l1_loss = tf.losses.absolute_difference(
            images, refine_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        refine_gen_loss += refine_gen_l1_loss

        # generator perceptual loss
        refine_content_x = self.vgg.forward(refine_outputs, reuse=True)
        refine_content_y = self.vgg.forward(images, reuse=True)
        refine_gen_content_loss = perceptual_loss(refine_content_x, refine_content_y) * self.cfg['CONTENT_LOSS_WEIGHT']
        refine_gen_loss += refine_gen_content_loss

        # generator style loss
        refine_style_x = self.vgg.forward(refine_outputs * masks, reuse=True)
        refine_style_y = self.vgg.forward(images * masks, reuse=True)
        refine_gen_style_loss = style_loss(refine_style_x, refine_style_y) * self.cfg['STYLE_LOSS_WEIGHT']
        refine_gen_loss += refine_gen_style_loss

        # get the refine model variables
        refine_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_generator')
        refine_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'refine_discriminator')

        # get the refine optimizers
        refine_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])
        refine_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])

        # optimize the refine models
        refine_gen_train = refine_gen_optimizer.minimize(refine_gen_loss,
                                                         global_step=gen_global_step,
                                                         var_list=refine_gen_vars)
        refine_dis_train = refine_dis_optimizer.minimize(refine_dis_loss,
                                                         global_step=dis_global_step,
                                                         var_list=refine_dis_vars)

        refine_dis_train_ops = []
        for i in range(5):
            refine_dis_train_ops.append(refine_dis_train)
        refine_dis_train = tf.group(*refine_dis_train_ops)

        # create logs
        refine_logs = [refine_dis_loss, refine_gen_loss, refine_gen_gan_loss,
                       refine_gen_l1_loss, refine_gen_style_loss, refine_gen_content_loss]

        # add summary for monitor
        tf.summary.scalar('refine_dis_loss', refine_dis_loss)
        tf.summary.scalar('refine_gen_loss', refine_gen_loss)
        tf.summary.scalar('refine_gen_gan_loss', refine_gen_gan_loss)
        tf.summary.scalar('refine_gen_l1_loss', refine_gen_l1_loss)
        tf.summary.scalar('refine_gen_style_loss', refine_gen_style_loss)
        tf.summary.scalar('refine_gen_content_loss', refine_gen_content_loss)

        refine_visual_img = [images, images_masked, coarse_outputs_merged, refine_outputs_merged]
        refine_visual_img = tf.concat(refine_visual_img, axis=2)
        images_summary(refine_visual_img, 'refine_gt_masked_coarse_refine', 4)

        refine_returned = [refine_outputs, refine_outputs_merged, refine_gen_train, refine_dis_train, refine_logs]

        # --------------------- Build joint loss function -----------------------------
        joint_gen_loss = 0.0
        joint_dis_loss = 0.0

        # discriminator loss
        joint_dis_input_real = images
        joint_dis_input_fake = tf.stop_gradient(refine_outputs)
        joint_dis_real, _ = self.refine_discriminator(joint_dis_input_real, reuse=True, use_sigmoid=use_sigmoid)
        joint_dis_fake, _ = self.refine_discriminator(joint_dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        joint_dis_real_loss = adversarial_loss(joint_dis_real, is_real=True,
                                               gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        joint_dis_fake_loss = adversarial_loss(joint_dis_fake, is_real=False,
                                               gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        joint_dis_loss += (joint_dis_real_loss + joint_dis_fake_loss) / 2.0

        # generator adversartial loss
        joint_gen_input_fake = refine_outputs
        joint_gen_fake, _ = self.refine_discriminator(joint_gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        joint_gen_gan_loss = adversarial_loss(joint_gen_fake, is_real=True,
                                              gan_type=self.cfg['GAN_LOSS'], is_disc=False)
        joint_gen_gan_loss *= self.cfg['ADV_LOSS_WEIGHT']
        joint_gen_loss += joint_gen_gan_loss

        # generator l1 loss
        joint_gen_l1_loss = tf.losses.absolute_difference(
            images, refine_outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        joint_gen_loss += joint_gen_l1_loss

        # generator perceptual loss
        joint_content_x = refine_content_x
        joint_content_y = refine_content_y
        joint_gen_content_loss = perceptual_loss(joint_content_x, joint_content_y) * self.cfg['CONTENT_LOSS_WEIGHT']
        joint_gen_loss += joint_gen_content_loss

        # generator style loss
        joint_style_x = refine_style_x
        joint_style_y = refine_style_y
        joint_gen_style_loss = style_loss(joint_style_x, joint_style_y) * self.cfg['STYLE_LOSS_WEIGHT']
        joint_gen_loss += joint_gen_style_loss

        joint_gen_vars = coarse_gen_vars + refine_gen_vars
        joint_dis_vars = refine_dis_vars

        # get the joint optimizers
        joint_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                     beta1=self.cfg['BETA1'],
                                                     beta2=self.cfg['BETA2'])
        joint_dis_optimizer = refine_dis_optimizer
        # joint_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
        #                                              beta1=self.cfg['BETA1'],
        #                                              beta2=self.cfg['BETA2'])

        # optimize the joint models
        joint_gen_train = joint_gen_optimizer.minimize(joint_gen_loss,
                                                       global_step=gen_global_step,
                                                       var_list=joint_gen_vars)
        joint_dis_train = joint_dis_optimizer.minimize(joint_dis_loss,
                                                       global_step=dis_global_step,
                                                       var_list=joint_dis_vars)

        joint_dis_train_ops = []
        for i in range(5):
            joint_dis_train_ops.append(joint_dis_train)
        joint_dis_train = tf.group(*joint_dis_train_ops)

        # create logs
        joint_logs = [joint_dis_loss, joint_gen_loss, joint_gen_gan_loss,
                      joint_gen_l1_loss, joint_gen_style_loss, joint_gen_content_loss]

        # add summary for monitor
        tf.summary.scalar('joint_dis_loss', joint_dis_loss)
        tf.summary.scalar('joint_gen_loss', joint_gen_loss)
        tf.summary.scalar('joint_gen_gan_loss', joint_gen_gan_loss)
        tf.summary.scalar('joint_gen_l1_loss', joint_gen_l1_loss)
        tf.summary.scalar('joint_gen_style_loss', joint_gen_style_loss)
        tf.summary.scalar('joint_gen_content_loss', joint_gen_content_loss)

        # add summaries for image visual
        joint_visual_img = [images, images_masked, coarse_outputs_merged, refine_outputs_merged]
        joint_visual_img = tf.concat(joint_visual_img, axis=2)
        images_summary(joint_visual_img, 'joint_gt_masked_coarse_refine', 4)

        joint_returned = [refine_outputs, refine_outputs_merged, joint_gen_train, joint_dis_train, joint_logs]

        return coarse_returned, refine_returned, joint_returned

    def save(self, sess, saver, path, model_name):
        print('\nsaving the model ...\n')
        saver.save(sess, os.path.join(path, model_name))

    def load(self, sess, saver, path, model_name):
        print('\nloading the model ...\n')
        saver.restore(sess, os.path.join(path, model_name))
