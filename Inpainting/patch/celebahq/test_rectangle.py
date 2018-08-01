from __future__ import print_function

import os
import platform
import numpy as np
import cv2
# import yaml
import tensorflow as tf
from model import CompletionModel


def bbox2mask_np(bbox, height, width):
    top, left, h, w = bbox
    mask = np.pad(array=np.ones((h, w)),
                  pad_width=((top, height - h - top), (left, width - w - left)),
                  mode='constant',
                  constant_values=0)
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, -1)
    mask = np.concatenate((mask, mask, mask), axis=3) * 255
    return mask


if platform.system() == 'Windows':
    prefix = 'F:\\Datasets\\celebahq'
    val_path = 'F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001'
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\celebahq\\model\\refine'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        prefix = '/home/icie/Datasets/celebahq'
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/celebahq/model/refine'

hole_size = 120
image_size = 256
bbox_np = ((image_size - hole_size) // 2,
           (image_size - hole_size) // 2,
           hole_size,
           hole_size)
mask = bbox2mask_np(bbox_np, image_size, image_size)

model = CompletionModel()
image_ph = tf.placeholder(tf.float32, (1, 256, 256, 3))
mask_ph = tf.placeholder(tf.float32, (1, 256, 256, 3))
inputs = tf.concat([image_ph, mask_ph], axis=2)
batch_incomplete, batch_complete_coarse, batch_complete_refine = model.build_test_graph(inputs)
batch_complete_refine = (batch_complete_refine + 1.) * 127.5
batch_complete_refine = tf.saturate_cast(batch_complete_refine, tf.uint8)

# metrics
# image value in (0,255)
ssim_tf = tf.image.ssim(tf.cast(image_ph[0], tf.uint8), batch_complete_refine[0], 255)
psnr_tf = tf.image.psnr(tf.cast(image_ph[0], tf.uint8), batch_complete_refine[0], 255)
tv_loss = tf.image.total_variation(image_ph[0]) -\
    tf.image.total_variation(tf.cast(batch_complete_refine[0], dtype=tf.float32))
tv_loss = tv_loss / tf.image.total_variation(image_ph[0])

# image value in (-1,1)
l1_loss = tf.reduce_mean(tf.abs(image_ph[0] -
                                tf.cast(batch_complete_refine[0], dtype=tf.float32))) / 127.5
l2_loss = tf.reduce_mean(tf.square(image_ph[0] -
                                   tf.cast(batch_complete_refine[0], dtype=tf.float32))) / 16256.25

ssims = []
psnrs = []
l1_losses = []
l2_losses = []
tv_losses = []

# saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    for i in range(1000):
        print('{}th image'.format(i + 1))
        img_path = os.path.join(prefix, 'img%.8d.png' % (i + 29000))
        # img_path = '2.jpeg'
        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        assert image.shape == mask.shape  # (1,256,256,3)
        # input_image = np.concatenate([image, mask], axis=2)

        result, ssim, psnr, l1, l2, tv = sess.run([batch_complete_refine, ssim_tf, psnr_tf, l1_loss, l2_loss, tv_loss],
                                                  feed_dict={image_ph: image, mask_ph: mask})

        ssims.append(ssim)
        psnrs.append(psnr)
        l1_losses.append(l1)
        l2_losses.append(l2)
        tv_losses.append(tv)

    cv2.imwrite('F:\\val.png', image.astype(np.uint8)[0])
    cv2.imwrite('F:\\output.png', result[0])

mean_ssim = np.mean(ssims)
mean_psnr = np.mean(psnrs)
mean_l1 = np.mean(l1_losses)
mean_l2 = np.mean(l2_losses)
mean_tv = np.mean(tv_losses)

print('ssim: {}'.format(mean_ssim))
print('psnr: {}'.format(mean_psnr))
print('l1_loss: {}'.format(mean_l1))
print('l2_loss: {}'.format(mean_l2))
print('tv_loss: {}'.format(mean_tv))
