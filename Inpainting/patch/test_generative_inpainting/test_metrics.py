import os
import platform
# import yaml
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


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


def bbox2mask(bbox, height, width):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns
    -------
        tf.Tensor: output with shape [1, H, W, 1]

    """
    # height = cfg['img_height']
    # width = cfg['img_width']
    top, left, h, w = bbox

    mask = tf.pad(tensor=tf.ones((h, w), dtype=tf.float32),
                  paddings=[[top, height - h - top],
                            [left, width - w - left]])

    # mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    mask = tf.concat([mask, mask, mask], axis=2)
    return mask


def parse_tfrecord(example_proto):
    features = {'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.uint8)
    # img = tf.reshape(data, parsed_features['shape'])
    img = tf.reshape(data, [1024, 1024, 3])

    return img


if platform.system() == 'Windows':
    prefix = 'F:\\Datasets\\celebahq'
    val_path = 'F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001'
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\test_generative_inpainting\\model_logs\\release_celebahq_256'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        prefix = '/home/icie/Datasets/celebahq'
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/test_generative_inpainting/model_logs/release_celebahq_256'

hole_size = 128
image_size = 256
bbox_np = ((image_size - hole_size) // 2,
           (image_size - hole_size) // 2,
           hole_size,
           hole_size)
# bbox_np = (54, 83, 80, 100)
# bbox_np = (148, 100, 84, 117)
mask = bbox2mask_np(bbox_np, image_size, image_size)

model = InpaintCAModel()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    ssims = []
    psnrs = []
    l1_losses = []
    l2_losses = []
    tv_losses = []
    for i in range(1000):
        print('{}th image'.format(i + 1))
        img_path = os.path.join(prefix, 'img%.8d.png' % (i + 29000))
        # img_path = '2.jpeg'
        image = cv2.imread(img_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, 0)
        assert image.shape == mask.shape  # (1,256,256,3)

        input_image = np.concatenate([image, mask], axis=2)
        input_image = tf.constant(input_image, dtype=tf.float32)
        if i == 0:
            output = model.build_server_graph(input_image)
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
        else:
            output = model.build_server_graph(input_image, reuse=True)

        output = (output + 1.) * 127.5
        output = tf.saturate_cast(output, tf.uint8)

        # image value in (0,255)
        ssim = tf.image.ssim(tf.constant(image[0]), output[0], 255)
        psnr = tf.image.psnr(tf.constant(image[0]), output[0], 255)
        tv_loss = tf.image.total_variation(tf.constant(image[0], dtype=tf.float32)) -\
            tf.image.total_variation(tf.cast(output[0], dtype=tf.float32))
        tv_loss = tv_loss / tf.image.total_variation(tf.constant(image[0], dtype=tf.float32))

        # image value in (-1,1)
        l1_loss = tf.reduce_mean(tf.abs(tf.constant(image[0], dtype=tf.float32) -
                                        tf.cast(output[0], dtype=tf.float32))) / 127.5
        l2_loss = tf.reduce_mean(tf.square(tf.constant(image[0], dtype=tf.float32) -
                                           tf.cast(output[0], dtype=tf.float32))) / 16256.25

        # result, ssim, psnr, l1, l2, tv = sess.run([output, ssim, psnr, l1_loss, l2_loss, tv_loss])
        ssims.append(ssim)
        psnrs.append(psnr)
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
        tv_losses.append(tv_loss)

    mean_ssim = tf.reduce_mean(ssims)
    mean_psnr = tf.reduce_mean(psnrs)
    mean_l1 = tf.reduce_mean(l1_losses)
    mean_l2 = tf.reduce_mean(l2_losses)
    mean_tv = tf.reduce_mean(tv_losses)

    mean_ssim, mean_psnr, mean_l1, mean_l2, mean_tv = sess.run([mean_ssim, mean_psnr, mean_l1, mean_l2, mean_tv])

    # cv2.imwrite('F:\\output.png', result[0])
    # cv2.imwrite('F:\\val.png', image[0])

    # mean_ssim = np.mean(ssims)
    # mean_psnr = np.mean(psnrs)
    # mean_l1 = np.mean(l1_losses)
    # mean_l2 = np.mean(l2_losses)
    # mean_tv = np.mean(tv_losses)
    print('ssim: {}'.format(mean_ssim))
    print('psnr: {}'.format(mean_psnr))
    print('l1_loss: {}'.format(mean_l1))
    print('l2_loss: {}'.format(mean_l2))
    print('tv_loss: {}'.format(mean_tv))
    # print(ssim, psnr, l1, l2, tv)
