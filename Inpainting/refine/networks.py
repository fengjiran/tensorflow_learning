from __future__ import print_function
import os
import tensorflow as tf

from ops import conv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss
from loss import perceptual_loss
# from loss import style_loss
from loss import Vgg19

from metrics import tf_l1_loss
from metrics import tf_l2_loss
from metrics import tf_psnr
from metrics import tf_ssim


class RefineNet():
    """Construct refine model."""

    def __init__(self, config=None):
        """Init method."""
        print('Construct the refine model.')
        self.cfg = config
        self.init_type = config['INIT_TYPE']
        self.vgg = Vgg19()

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

            return edge, x

    def color_domain_generator(self, x, reuse=None):
        with tf.variable_scope('color_generator', reuse=reuse):
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

            # x = deconv(x, channels=128, kernel=4, stride=2, init_type=self.init_type, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            shape2 = tf.shape(x)
            x = tf.image.resize_nearest_neighbor(x, size=(shape2[1] * 2, shape2[2] * 2))
            x = conv(x, channels=64, kernel=3, stride=1, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv5')

            # x = deconv(x, channels=64, kernel=4, stride=2, init_type=self.init_type, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv6')

            x = tf.nn.sigmoid(x)

            return x

    def inpaint_generator(self, x, reuse=None):
        with tf.variable_scope('inpaint_generator', reuse=reuse):
            color_domains = x[:, :, :, 3:6]
            edges = x[:, :, :, 6:7]
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
            color = tf.image.resize_nearest_neighbor(color_domains, size=(shape1[1] * 2, shape1[2] * 2))
            edge = tf.image.resize_nearest_neighbor(edges, size=(shape1[1] * 2, shape1[2] * 2))
            edge = tf.cast(tf.greater(edge, 0.25), dtype=tf.float32)
            x = tf.concat([x, color, edge], axis=3)

            x = conv(x, channels=128, kernel=3, stride=1, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv4')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            shape2 = tf.shape(x)
            x = tf.image.resize_nearest_neighbor(x, size=(shape2[1] * 2, shape2[2] * 2))
            color = tf.image.resize_nearest_neighbor(color_domains, size=(shape2[1] * 2, shape2[2] * 2))
            edge = tf.image.resize_nearest_neighbor(edges, size=(shape2[1] * 2, shape2[2] * 2))
            edge = tf.cast(tf.greater(edge, 0.25), dtype=tf.float32)
            x = tf.concat([x, color, edge], axis=3)

            x = conv(x, channels=64, kernel=3, stride=1, pad=1,
                     pad_type='reflect', init_type=self.init_type, name='conv5')
            # x = deconv(x, channels=64, kernel=4, stride=2, init_type=self.init_type, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            color = tf.image.resize_nearest_neighbor(color_domains, size=(
                self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']))
            edge = tf.image.resize_nearest_neighbor(edges, size=(self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE']))
            edge = tf.cast(tf.greater(edge, 0.25), dtype=tf.float32)
            x = tf.concat([x, color, edge], axis=3)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv6')

            x = tf.nn.tanh(x)

            return x

    def inpaint_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('inpaint_discriminator', reuse=reuse):
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

    def build_model(self, images, img_grays, edges, color_domains, masks):
        edges_masked = edges * (1 - masks)
        grays_masked = img_grays * (1 - masks) + masks
        edge_inputs = tf.concat([grays_masked, edges_masked, masks * tf.ones_like(img_grays)], axis=3)
        edge_outputs, edge_logits = self.edge_generator(edge_inputs)
        edge_outputs_merged = edge_outputs * masks + edges * (1 - masks)

        color_domains_masked = color_domains * (1 - masks) + masks
        imgs_masked = images * (1 - masks) + masks
        color_inputs = tf.concat([imgs_masked, color_domains_masked,
                                  masks * tf.ones_like(images[:, :, :, 0:1])], axis=3)
        color_outputs = self.color_domain_generator(color_inputs)
        color_outputs_merged = color_outputs * masks + color_domains * (1 - masks)

        refine_inputs = tf.concat([imgs_masked, color_outputs_merged, edge_outputs_merged,
                                   masks * tf.ones_like(images[:, :, :, 0:1])], axis=3)
        outputs = self.inpaint_generator(refine_inputs)
        outputs_merged = outputs * masks + images * (1 - masks)

        # metrics
        psnr = tf_psnr(images, outputs_merged, 2.0)
        ssim = tf_ssim(images, outputs_merged, 2.0)
        l1 = tf_l1_loss(images, outputs_merged)
        l2 = tf_l2_loss(images, outputs_merged)

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
        dis_input_real = images
        dis_input_fake = tf.stop_gradient(outputs_merged)
        dis_real, dis_real_feat = self.inpaint_discriminator(dis_input_real, use_sigmoid=use_sigmoid)
        dis_fake, dis_fake_feat = self.inpaint_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True,
                                         gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False,
                                         gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_fake_loss + dis_real_loss) / 2.0

        # generator l1 loss
        gen_l1_loss = tf.losses.absolute_difference(images, outputs) / tf.reduce_mean(masks)

        # generator adversarial loss
        gen_input_fake = outputs_merged
        gen_fake, gen_fake_feat = self.inpaint_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True,
                                        gan_type=self.cfg['GAN_LOSS'], is_disc=False)

        # generator perceptual loss
        content_x = self.vgg.forward(outputs_merged)
        content_y = self.vgg.forward(images, reuse=True)
        gen_content_loss = perceptual_loss(content_x, content_y)

        gen_loss = gen_l1_loss * self.cfg['L1_LOSS_WEIGHT'] + \
            gen_gan_loss * self.cfg['ADV_LOSS_WEIGHT'] + \
            gen_content_loss * self.cfg['CONTENT_LOSS_WEIGHT']

        # get model variables
        edge_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'edge_generator')
        color_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'color_generator')
        inpaint_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_generator')
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_discriminator')

        gen_vars = edge_gen_vars + color_gen_vars + inpaint_gen_vars

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
        for i in range(5):
            dis_train_ops.append(dis_train)
        dis_train = tf.group(*dis_train_ops)

        # create logs
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_content_loss, psnr, ssim, l1, l2]

        # add summary for monitor
        tf.summary.scalar('dis_loss', dis_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss)
        tf.summary.scalar('gen_content_loss', gen_content_loss)
        # tf.summary.scalar('gen_style_loss', gen_style_loss)

        tf.summary.scalar('train_psnr', psnr)
        tf.summary.scalar('train_ssim', ssim)
        tf.summary.scalar('train_l1', l1)
        tf.summary.scalar('train_l2', l2)

        return gen_train, dis_train, logs
