from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss
from loss import perceptual_loss
from loss import style_loss

from utils import images_summary


class InpaintingModel():
    """Construct model."""

    def __init__(self, config=None):
        print('Construct the inpainting model.')
        self.cfg = config
        self.init_type = self.cfg['INIT_TYPE']

    def coarse_generator(self, x):
        with tf.variable_scope('coarse_generator'):
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

    def build_coarse_model(self, images, masks):
        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks
        inputs = tf.concat([images_masked, masks], axis=3)
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
            gen_fake, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=False) * self.cfg['COARSE_ADV_LOSS_WEIGHT']
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = tf.losses.absolute_difference(
            images, outputs) * self.cfg['L1_LOSS_WEIGHT'] / tf.reduce_mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = perceptual_loss(outputs, images) * self.cfg['CONTENT_LOSS_WEIGHT']
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = style_loss(outputs * masks, images * masks) * self.cfg['STYLE_LOSS_WEIGHT']
        gen_loss += gen_style_loss

        coarse_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_generator')
        coarse_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'coarse_generator')

        coarse_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])
        coarse_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                      beta1=self.cfg['BETA1'],
                                                      beta2=self.cfg['BETA2'])

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

        coarse_gen_train = coarse_gen_optimizer.minimize(gen_loss,
                                                         global_step=coarse_gen_global_step,
                                                         var_list=coarse_gen_vars)
        coarse_dis_train = coarse_dis_optimizer.minimize(dis_loss,
                                                         global_step=coarse_dis_global_step,
                                                         var_list=coarse_dis_vars)

        visual_img = [images, images_masked, outputs_merged]
        visual_img = tf.concat(visual_img, axis=2)
        images_summary(visual_img, 'groundtruth_masked_inpainted', 4)

        return outputs, outputs_merged, gen_loss, dis_loss, coarse_gen_train, coarse_dis_train
