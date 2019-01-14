from __future__ import print_function
import tensorflow as tf

from ops import conv
from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss


class InpaintingModel(object):
    """Construct model."""

    def __init__(self, config=None):
        print('Construct the inpainting model.')
        self.cfg = config

    def edge_generator(self, x):
        with tf.variable_scope('edge_generator'):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', name='conv3')
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
            x = deconv(x, channels=128, kernel=4, stride=2, name='deconv1')
            x = instance_norm(x, name='in4')
            x = tf.nn.relu(x)

            x = deconv(x, channels=64, kernel=4, stride=2, name='deconv2')
            x = instance_norm(x, name='in5')
            x = tf.nn.relu(x)

            x = conv(x, channels=1, kernel=7, stride=1, pad=3, pad_type='reflect', name='conv4')
            x = tf.nn.sigmoid(x)

            return x

    def edge_discriminator(self, x, reuse=None, use_sigmoid=False):
        with tf.variable_scope('edge_discriminator', reuse=reuse):
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

    def inpaint_generator(self, x):
        with tf.variable_scope('inpaint_generator'):
            # encoder
            x = conv(x, channels=64, kernel=7, stride=1, pad=3, pad_type='reflect', sn=False, name='conv1')
            x = instance_norm(x, name='in1')
            x = tf.nn.relu(x)

            x = conv(x, channels=128, kernel=4, stride=2, pad=1, pad_type='zero', sn=False, name='conv2')
            x = instance_norm(x, name='in2')
            x = tf.nn.relu(x)

            x = conv(x, channels=256, kernel=4, stride=2, pad=1, pad_type='zero', sn=False, name='conv3')
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
        edges_masked = edges * (1 - masks)
        images_masked = images * (1 - masks) + masks

        # in: [grayscale(1)+edge(1)+mask(1)]
        inputs = tf.concat([images_masked, edges_masked, masks], axis=3)

        outputs = self.edge_generator(inputs)

        dis_loss = 0.0
        gen_loss = 0.0

        # discriminator loss
        dis_input_real = tf.concat([images, edges], axis=3)
        dis_input_fake = tf.concat([images, outputs], axis=3)

        if self.cfg['GAN_LOSS'] == 'lsgan':
            use_sigmoid = True
        else:
            use_sigmoid = False

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
        for i in range(len(dis_real_features)):
            pass


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
