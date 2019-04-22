from __future__ import print_function
import os
import tensorflow as tf

from ops import conv
# from ops import deconv
from ops import resnet_block
from ops import instance_norm

from loss import adversarial_loss
from loss import perceptual_loss
from loss import style_loss
from loss import Vgg19

from metrics import tf_l1_loss
from metrics import tf_l2_loss
from metrics import tf_psnr
from metrics import tf_ssim


class InpaintModel():
    """Construct inpaint model."""

    def __init__(self, config=None):
        print('Construct the inpaint model.')
        self.cfg = config
        self.init_type = self.cfg['INIT_TYPE']
        self.vgg = Vgg19()

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

    def build_model(self, images, edges, color_domains, masks):
        # generator input: [img_masked(3) + edge(1) + color_domain(3) + mask(1)]
        # discriminator input: [img(3)]
        imgs_masked = images * (1 - masks) + masks
        inputs = tf.concat([imgs_masked, color_domains, edges,
                            masks * tf.ones_like(tf.expand_dims(images[:, :, :, 0], -1))], axis=3)
        outputs = self.inpaint_generator(inputs)
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

        # generator style loss
        style_x = self.vgg.forward(outputs_merged * masks, reuse=True)
        style_y = self.vgg.forward(images * masks, reuse=True)
        gen_style_loss = style_loss(style_x, style_y)

        gen_loss = gen_l1_loss * self.cfg['L1_LOSS_WEIGHT'] + \
            gen_gan_loss * self.cfg['ADV_LOSS_WEIGHT'] + \
            gen_content_loss * self.cfg['CONTENT_LOSS_WEIGHT'] +\
            gen_style_loss * self.cfg['STYLE_LOSS_WEIGHT']

        # get model variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_generator')
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_discriminator')

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
        logs = [dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, psnr, ssim, l1, l2]

        # add summary for monitor
        tf.summary.scalar('dis_loss', dis_loss)
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss)
        tf.summary.scalar('gen_content_loss', gen_content_loss)
        tf.summary.scalar('gen_style_loss', gen_style_loss)

        tf.summary.scalar('train_psnr', psnr)
        tf.summary.scalar('train_ssim', ssim)
        tf.summary.scalar('train_l1', l1)
        tf.summary.scalar('train_l2', l2)

        return gen_train, dis_train, logs

    def eval_model(self, images, edges, color_domains, masks):
        # generator input: [img_masked(3) + edge(1) + color_domain(3) + mask(1)]
        imgs_masked = images * (1 - masks) + masks
        inputs = tf.concat([imgs_masked, color_domains, edges,
                            masks * tf.ones_like(tf.expand_dims(images[:, :, :, 0], -1))], axis=3)
        outputs = self.inpaint_generator(inputs, reuse=True)
        outputs_merged = outputs * masks + images * (1 - masks)

        # metrics
        psnr = tf_psnr(images, outputs_merged, 2.0)
        ssim = tf_ssim(images, outputs_merged, 2.0)
        l1 = tf_l1_loss(images, outputs_merged)
        l2 = tf_l2_loss(images, outputs_merged)

        tf.summary.scalar('train_psnr', psnr)
        tf.summary.scalar('train_ssim', ssim)
        tf.summary.scalar('train_l1', l1)
        tf.summary.scalar('train_l2', l2)

        visual_img = [images, color_domains, imgs_masked, outputs_merged]
        visual_img = tf.concat(visual_img, axis=2)
        tf.summary.image('image_edge_color_merge', visual_img, 5)

        val_logs = [psnr, ssim, l1, l2]

        return val_logs

    def test_model(self, images, edges, color_domains, masks):
        # generator input: [img_masked(3) + edge(1) + color_domain(3) + mask(1)]
        imgs_masked = images * (1 - masks) + masks
        inputs = tf.concat([imgs_masked, color_domains, edges,
                            masks * tf.ones_like(tf.expand_dims(images[:, :, :, 0], -1))], axis=3)
        outputs = self.inpaint_generator(inputs)
        outputs_merged = outputs * masks + images * (1 - masks)
        return outputs_merged

    def save(self, sess, saver, path, model_name):
        print('\nsaving the model...\n')
        saver.save(sess, os.path.join(path, model_name))

    def load(self, sess, saver, path, model_name):
        print('\nloading the model...\n')
        saver.restore(sess, os.path.join(path, model_name))


if __name__ == '__main__':
    import os
    import yaml
    import numpy as np
    import platform as pf
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    if pf.system() == 'Windows':
        vgg19_npy_path = 'F:\\Datasets\\vgg19.npy'
        log_dir = cfg['LOG_DIR_WIN']
        model_dir = cfg['MODEL_PATH_WIN']
        train_flist = cfg['TRAIN_FLIST_WIN']
        val_flist = cfg['VAL_FLIST_WIN']
        test_flist = cfg['TEST_FLIST_WIN']
        mask_flist = cfg['MASK_FLIST_WIN']
    elif pf.system() == 'Linux':
        if pf.node() == 'icie-Precision-Tower-7810':
            vgg19_npy_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/vgg19.npy'
            log_dir = cfg['LOG_DIR_LINUX_7810']
            model_dir = cfg['MODEL_PATH_LINUX_7810']
            train_flist = cfg['TRAIN_FLIST_LINUX_7810']
            val_flist = cfg['VAL_FLIST_LINUX_7810']
            test_flist = cfg['TEST_FLIST_LINUX_7810']
            mask_flist = cfg['MASK_FLIST_LINUX_7810']
        elif pf.node() == 'icie-Precision-T7610':
            vgg19_npy_path = '/home/icie/Datasets/vgg19.npy'
            log_dir = cfg['LOG_DIR_LINUX_7610']
            model_dir = cfg['MODEL_PATH_LINUX_7610']
            train_flist = cfg['TRAIN_FLIST_LINUX_7610']
            val_flist = cfg['VAL_FLIST_LINUX_7610']
            test_flist = cfg['TEST_FLIST_LINUX_7610']
            mask_flist = cfg['MASK_FLIST_LINUX_7610']

    model = InpaintModel(cfg)

    images = tf.placeholder(tf.float32, [10, 256, 256, 3])
    color_domains = tf.placeholder(tf.float32, [10, 256, 256, 3])
    edges = tf.placeholder(tf.float32, [10, 256, 256, 1])
    masks = tf.placeholder(tf.float32, [10, 256, 256, 1])

    gen, dis, log = model.build_model(images, edges, color_domains, masks)

    # gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_generator')
    # dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_discriminator')

    # var_list = gen_vars + dis_vars

    # data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg')
    for var in vgg_var_list:
        var_list.remove(var)

    # assign_ops = []
    # for var in vgg_var_list:
    #     vname = var.name
    #     vname = vname.split('/')[1]
    #     var_shape = var.get_shape().as_list()
    #     if len(var_shape) != 1:
    #         var_value = data_dict[vname][0]
    #     else:
    #         var_value = data_dict[vname][1]
    #     assign_ops.append(tf.assign(var, var_value))

    saver = tf.train.Saver(var_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(model_dir, 'model'))
        # saver.restore(sess, os.path.join(model_dir, 'model'))
