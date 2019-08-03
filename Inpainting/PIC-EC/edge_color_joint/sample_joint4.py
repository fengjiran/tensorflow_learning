import os
import glob
import yaml
import numpy as np
import cv2
from imageio import imread
from imageio import imwrite
from skimage.feature import canny
from skimage.color import rgb2gray
import tensorflow as tf

from ops import conv
from ops import resnet_block
from ops import instance_norm
from loss import Vgg19
from utils import get_color_domain


class JointNetTest():
    """Construct refine model."""

    def __init__(self, config=None):
        """Init method."""
        print('Construct the refine model.')
        self.cfg = config
        self.init_type = cfg['INIT_TYPE']
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
            # edge = tf.nn.sigmoid(x)
            edge = tf.nn.tanh(x)
            edge = (edge + 1) / 2.

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

    def sample(self, images, edges, color_domains, masks):
        color_domains_masked = color_domains * (1 - masks) + masks
        imgs_masked = images * (1 - masks) + masks
        color_inputs = tf.concat([imgs_masked, color_domains_masked,
                                  masks * tf.ones_like(images[:, :, :, 0:1])], axis=3)
        color_outputs = self.color_domain_generator(color_inputs)
        color_outputs_merged = color_outputs * masks + color_domains * (1 - masks)

        refine_inputs = tf.concat([imgs_masked, color_outputs_merged, edges,
                                   masks * tf.ones_like(images[:, :, :, 0:1])], axis=3)
        outputs = self.inpaint_generator(refine_inputs)
        outputs_merged = outputs * masks + images * (1 - masks)

        return outputs_merged


def load_mask(cfg, mask_type=1, mask_path=None):
    if mask_type == 1:  # random block
        hole_size = cfg['INPUT_SIZE'] // 2
        top = np.random.randint(0, hole_size + 1)
        left = np.random.randint(0, hole_size + 1)
        img_mask = np.pad(array=np.ones((hole_size, hole_size)),
                          pad_width=((top, cfg['INPUT_SIZE'] - hole_size - top),
                                     (left, cfg['INPUT_SIZE'] - hole_size - left)),
                          mode='constant',
                          constant_values=0)

        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)  # (1, 256, 256, 1) float
    else:  # external
        img_mask = imread(mask_path)
        # img_mask = cv2.resize(img_mask, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)
        img_mask = img_mask > 3
        img_mask = img_mask.astype(np.float32)
        img_mask = np.expand_dims(img_mask, 0)
        img_mask = np.expand_dims(img_mask, -1)
        img_mask = 1 - img_mask

    return img_mask  # (1, 256, 256, 1) float


def load_items(cfg, image_path):
    image = imread(image_path)  # [1024, 1024, 3], [0, 255]
    # image = cv2.resize(image, (cfg['INPUT_SIZE'], cfg['INPUT_SIZE']), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
    gray = rgb2gray(image)
    color = get_color_domain(image, cfg['BLUR_FACTOR1'], cfg['BLUR_FACTOR1'], cfg['K'])

    edge = canny(gray, sigma=cfg['SIGMA'])
    edge = edge.astype(np.float32)

    image = np.expand_dims(image, 0)  # (1, 256, 256, 3)
    color = np.expand_dims(color, 0)  # (1, 256, 256, 3)

    edge = np.expand_dims(edge, 0)
    edge = np.expand_dims(edge, -1)

    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    image = image / 127.5 - 1

    return image, gray, color, edge


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            # return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            try:
                print('is a file')
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]

    return []


if __name__ == '__main__':
    with open('test_joint_flag.yaml', 'r') as f:
        cfg_flag = yaml.load(f, Loader=yaml.FullLoader)
        flag = cfg_flag['flag']

    if flag == 1:
        cfg_name = 'test_joint_celeba_regular.yaml'
    elif flag == 2:
        cfg_name = 'test_joint_celeba_irregular.yaml'
    elif flag == 3:
        cfg_name = 'test_joint_celebahq_regular.yaml'
    elif flag == 4:
        cfg_name = 'test_joint_celebahq_irregular.yaml'
    elif flag == 5:
        cfg_name = 'test_joint_psv_regular.yaml'
    elif flag == 6:
        cfg_name = 'test_joint_psv_irregular.yaml'
    elif flag == 7:
        cfg_name = 'test_joint_places2_regular.yaml'
    elif flag == 8:
        cfg_name = 'test_joint_places2_irregular.yaml'

    with open(cfg_name, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    edge_checkpoint_dir = cfg['EDGE_MODEL_PATH']
    color_checkpoint_dir = cfg['COLOR_MODEL_PATH']
    joint_checkpoint_dir = cfg['JOINT_MODEL_PATH']

    sample_dir = cfg['SAMPLE_DIR']
    mask_type = cfg['MASK']
    mask_paths = load_flist(cfg['TEST_MASK_PATH'])
    image_paths = load_flist(cfg['TEST_IMAGE_PATH'])

    print(mask_paths[0], mask_paths[1])
    print(image_paths[0], image_paths[1])


########################### construct the model ##################################
model = JointNetTest(cfg)
# 1 for missing region, 0 for background
mask = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
# gray = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
edge = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 1])
image = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
color = tf.placeholder(tf.float32, [1, cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
output = model.sample(image, edge, color, mask)
##################################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # edge_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'edge_generator')
    color_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'color_generator')
    inpaint_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_generator')

    # edge_assign_ops = []
    color_assign_ops = []
    inpaint_assign_ops = []

    # for var in edge_gen_vars:
    #     vname = var.name
    #     from_name = vname
    #     var_value = tf.train.load_variable(os.path.join(edge_checkpoint_dir, 'model'), from_name)
    #     edge_assign_ops.append(tf.assign(var, var_value))
    # sess.run(edge_assign_ops)
    # print('Edge Model loaded.')

    for var in color_gen_vars:
        vname = var.name
        from_name = vname
        var_value = tf.train.load_variable(os.path.join(color_checkpoint_dir, 'model'), from_name)
        color_assign_ops.append(tf.assign(var, var_value))
    sess.run(color_assign_ops)
    print('Color Model loaded.')

    for var in inpaint_gen_vars:
        vname = var.name
        from_name = vname
        var_value = tf.train.load_variable(os.path.join(joint_checkpoint_dir, 'model'), from_name)
        inpaint_assign_ops.append(tf.assign(var, var_value))
    sess.run(inpaint_assign_ops)
    print('Joint Model loaded.')

    i = 0
    for (img_path, mask_path) in zip(image_paths, mask_paths):
        i = i + 1
        print(i)
        img_mask = load_mask(cfg, mask_type, mask_path)
        img, img_gray, img_color, img_edge = load_items(cfg, img_path)
        feed_dict = {image: img, color: img_color, edge: img_edge, mask: img_mask}

        inpainted_image = sess.run(output, feed_dict=feed_dict)
        inpainted_image = np.reshape(inpainted_image, [cfg['INPUT_SIZE'], cfg['INPUT_SIZE'], 3])
        imwrite(os.path.join(sample_dir, 'test_img_%04d_fake.png' % i), inpainted_image)
        # imwrite(os.path.join(sample_dir, 'psv_regular_inpainted_%03d.png' % i), inpainted_image)
