from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss
from loss import perceptual_loss
from loss import style_loss


class InpaintingModel(object):
    """Construct model."""

    def __init__(self, config=None):
        print('Construct the inpainting model.')
        self.cfg = config
        self.edge_gen_optimizer = None
        self.edge_dis_optimizer = None
        self.inpaint_gen_optimizer = None
        self.inpaint_dis_optimizer = None

        self.init_type = self.cfg['INIT_TYPE']

    def edge_generator(self, x):
        with tf.variable_scope('edge_generator'):
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

            x = conv(x, channels=1, kernel=7, stride=1, pad=3,
                     pad_type='reflect', init_type=self.init_type, name='conv4')
            x = tf.nn.sigmoid(x)

            return x

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

    def inpaint_generator(self, x):
        with tf.variable_scope('inpaint_generator'):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3, pad_type='reflect',
                     sn=False, init_type=self.init_type, name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero',
                     sn=False, init_type=self.init_type, name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero',
                     sn=False, init_type=self.init_type, name='conv3')
            x = instance_norm(x, name='in3')
            x = tf.nn.relu(x)

            # resnet block
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block1')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block2')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block3')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block4')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block5')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block6')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block7')
            x = resnet_block(x, out_channels=256, dilation=2, name='resnet_block8')

            # decoder
            x = deconv(x, channels=128, kernel=4, stride=2, sn=False, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            x = deconv(x, channels=64, kernel=4, stride=2, sn=False, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv4')

            x = (tf.nn.tanh(x) + 1.) / 2.

            return x

    def inpaint_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('inpaint_discriminator', reuse=reuse):
            conv1 = conv(x, channels=64, kernel=4, stride=2, pad=1, pad_type='zero', use_bias=False, name='conv1')
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = conv(conv1, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', use_bias=False, name='conv2')
            conv2 = tf.nn.leaky_relu(conv2)

            conv3 = conv(conv2, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', use_bias=False, name='conv3')
            conv3 = tf.nn.leaky_relu(conv3)

            conv4 = conv(conv3, channels=512, kernel=4, stride=1, pad=1, pad_type='zero', use_bias=False, name='conv4')
            conv4 = tf.nn.leaky_relu(conv4)

            conv5 = conv(conv4, channels=1, kernel=4, stride=1, pad=1, pad_type='zero', use_bias=False, name='conv5')

            outputs = conv5
            if use_sigmoid:
                outputs = tf.nn.sigmoid(conv5)

            return outputs, [conv1, conv2, conv3, conv4, conv5]

    def build_edge_model(self, images, edges, masks):
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: [grayscale(1) + edge(1)]
        edges_masked = edges * (1.0 - masks)
        images_masked = images * (1.0 - masks) + masks

        # in: [grayscale(1)+edge(1)+mask(1)]
        inputs = tf.concat([images_masked, edges_masked, masks], axis=3)
        outputs = self.edge_generator(inputs)

        dis_loss = 0.0
        gen_loss = 0.0

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # discriminator loss
        dis_input_real = tf.concat([images, edges], axis=3)
        dis_input_fake = tf.concat([images, tf.stop_gradient(outputs)], axis=3)
        # in: [grayscale(1) + edge(1)]
        dis_real, dis_real_features = self.edge_discriminator(dis_input_real, use_sigmoid=use_sigmoid)
        # in: [grayscale(1) + edge(1)]
        dis_fake, dis_fake_features = self.edge_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        dis_real_loss = adversarial_loss(dis_real, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2.0

        # generator adversarial loss
        gen_input_fake = tf.concat([images, outputs], axis=3)
        gen_fake, gen_fake_features = self.edge_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True, is_disc=False)
        gen_loss += gen_gan_loss

        # generator features matching loss
        gen_fm_loss = 0.0
        for (a, b) in zip(gen_fake_features, dis_real_features):
            gen_fm_loss += tf.losses.absolute_difference(a, tf.stop_gradient(b))
        # for i in range(len(dis_real_features)):
        #     gen_fm_loss += tf.losses.absolute_difference(gen_fake_features[i], dis_real_features[i])

        gen_fm_loss = gen_fm_loss * self.cfg['FM_LOSS_WEIGHT']
        gen_loss += gen_fm_loss

        edge_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'edge_generator')
        edge_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'edge_discriminator')

        self.edge_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                         beta1=self.cfg['BETA1'],
                                                         beta2=self.cfg['BETA2'])
        self.edge_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                         beta1=self.cfg['BETA1'],
                                                         beta2=self.cfg['BETA2'])

        edge_gen_global_step = tf.get_variable('edge_gen_global_step',
                                               [],
                                               tf.int32,
                                               initializer=tf.zeros_initializer(),
                                               trainable=False)
        edge_dis_global_step = tf.get_variable('edge_dis_global_step',
                                               [],
                                               tf.int32,
                                               initializer=tf.zeros_initializer(),
                                               trainable=False)

        edge_gen_train = self.edge_gen_optimizer.minimize(gen_loss,
                                                          global_step=edge_gen_global_step,
                                                          var_list=edge_gen_vars)

        edge_dis_train = self.edge_dis_optimizer.minimize(dis_loss,
                                                          global_step=edge_dis_global_step,
                                                          var_list=edge_dis_vars)

        return outputs, gen_loss, dis_loss, edge_gen_train, edge_dis_train

    def build_inpaint_model(self, images, edges, masks):
        # generator input: [rgb(3)+edge(1)]
        # discriminator input: [rgb(3)]
        images_masked = images * (1.0 - masks) + masks
        inputs = tf.concat([images_masked, edges], axis=3)

        outputs = self.inpaint_generator(inputs)  # in: [rgb(3)+edge(1)]

        dis_loss = 0.0
        gen_loss = 0.0

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

        # discriminator loss
        dis_input_real = images
        dis_input_fake = tf.stop_gradient(outputs)
        dis_real, _ = self.inpaint_discriminator(dis_input_real, use_sigmoid=use_sigmoid)  # in: [rgb(3)]
        dis_fake, _ = self.inpaint_discriminator(dis_input_fake, reuse=True, use_sigmoid=use_sigmoid)  # in: [rgb(3)]
        dis_real_loss = adversarial_loss(dis_real, is_real=True, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_fake_loss = adversarial_loss(dis_fake, is_real=False, gan_type=self.cfg['GAN_LOSS'], is_disc=True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2.0

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.inpaint_discriminator(gen_input_fake, reuse=True, use_sigmoid=use_sigmoid)  # in: [rgb(3)]
        gen_gan_loss = adversarial_loss(gen_fake, is_real=True, is_disc=False) * self.cfg['INPAINT_ADV_LOSS_WEIGHT']
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

        inpaint_gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_generator')
        inpaint_dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_discriminator')

        self.inpaint_gen_optimizer = tf.train.AdamOptimizer(self.cfg['LR'],
                                                            beta1=self.cfg['BETA1'],
                                                            beta2=self.cfg['BETA2'])
        self.inpaint_dis_optimizer = tf.train.AdamOptimizer(self.cfg['LR'] * self.cfg['D2G_LR'],
                                                            beta1=self.cfg['BETA1'],
                                                            beta2=self.cfg['BETA2'])

        inpaint_gen_global_step = tf.get_variable('inpaint_gen_global_step',
                                                  [],
                                                  tf.int32,
                                                  initializer=tf.zeros_initializer(),
                                                  trainable=False)
        inpaint_dis_global_step = tf.get_variable('inpaint_dis_global_step',
                                                  [],
                                                  tf.int32,
                                                  initializer=tf.zeros_initializer(),
                                                  trainable=False)

        inpaint_gen_train = self.inpaint_gen_optimizer.minimize(gen_loss,
                                                                global_step=inpaint_gen_global_step,
                                                                var_list=inpaint_gen_vars)
        inpaint_dis_train = self.inpaint_dis_optimizer.minimize(dis_loss,
                                                                global_step=inpaint_dis_global_step,
                                                                var_list=inpaint_dis_vars)

        return outputs, gen_loss, dis_loss, inpaint_gen_train, inpaint_dis_train


if __name__ == '__main__':
    model = InpaintingModel()

    bs = 10
    x = tf.random_uniform([bs, 256, 256, 3])

    out = model.edge_generator(x)
    dis_out, dis_mid = model.edge_discriminator(x)
    inpaint_out = model.inpaint_generator(x)
    print(out.get_shape())
    print(dis_out.get_shape())
    print(inpaint_out.get_shape())
    # tf.keras.applications.vgg19()
