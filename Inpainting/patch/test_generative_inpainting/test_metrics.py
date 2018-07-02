# import os
import platform
# import yaml
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


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
    val_path = 'F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001'
    checkpoint_dir = 'E:\\TensorFlow_Learning\\Inpainting\\patch\\test_generative_inpainting\\model_logs\\release_celebahq_256'
elif platform.system() == 'Linux':
    if platform.node() == 'icie-Precision-T7610':
        val_path = '/home/icie/Datasets/celebahq_tfrecords/val/celebahq_valset.tfrecord-001'
        checkpoint_dir = '/home/richard/TensorFlow_Learning/Inpainting/patch/test_generative_inpainting/model_logs/release_celebahq_256'

val_filenames = tf.placeholder(tf.string)
val_data = tf.data.TFRecordDataset(val_filenames)
val_data = val_data.map(parse_tfrecord)
val_data = val_data.batch(1)
# val_data = val_data.repeat()
val_iterator = val_data.make_initializable_iterator()
val_batch_data = val_iterator.get_next()
val_batch_data = tf.image.resize_area(val_batch_data, [256, 256])
# val_batch_data = tf.clip_by_value(val_batch_data, 0., 255.)
# val_batch_data = val_batch_data / 127.5 - 1
# val_batch_data = tf.reshape(val_batch_data, [256, 256, 3])

hole_size = 128
image_size = 256
bbox = (tf.constant((image_size - hole_size) // 2),
        tf.constant((image_size - hole_size) // 2),
        tf.constant(hole_size),
        tf.constant(hole_size))

mask = bbox2mask(bbox, image_size, image_size) * 255  # (256,256,3)
mask = tf.expand_dims(mask, 0)  # (1,256,256,3)
# print(mask.get_shape())
# print(val_batch_data.get_shape())

input_image = tf.concat([val_batch_data, mask], axis=2)  # (1,256,512,3)

# print(input_image.get_shape())
# ng.get_gpus(1)
# args = parser.parse_args()
model = InpaintCAModel()
# output = model.build_server_graph(input_image)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(val_iterator.initializer, feed_dict={val_filenames: val_path})
    input_image = sess.run(input_image)
    # val = sess.run(val_batch_data)
    # val = (val + 1) * 127.5
    # val = np.reshape(val, (256, 256, 3))
    # val = val.astype(np.uint8)
    # cv2.imwrite('F:\\val.png', val[:, :, ::-1])

    input_image = tf.constant(input_image, dtype=tf.float32)
    output = model.build_server_graph(input_image)  # (-1, 1)
    output = (output + 1.) * 127.5  # (0, 255)
    output = tf.saturate_cast(output, tf.uint8)

    ssim = tf.image.ssim(tf.saturate_cast(val_batch_data[0], tf.uint8), output[0], 255)
    psnr = tf.image.psnr(tf.saturate_cast(val_batch_data[0], tf.uint8), output[0], 255)
    tmp1 = val_batch_data[0]  # / 127.5 - 1
    tmp2 = tf.cast(output[0], tf.float32)  # / 127.5 - 1
    l1_loss = tf.reduce_mean(tf.abs(tmp1 - tmp2))
    l2_loss = tf.reduce_mean(tf.square(tmp1 - tmp2))

    output = tf.reverse(output, [-1])

    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    result = sess.run(output)
    # print(result.min())
    # print(result.max())
    # print(result.shape)
    # result = np.reshape(result, (256, 256, 3))
    # result = (result + 1) * 127.5
    # resule = result.astype(np.uint8)
    # print(result.max())
    cv2.imwrite('F:\\output.png', result[0])

    show_ssim, show_psnr, show_l1, show_l2 = sess.run([ssim, psnr, l1_loss, l2_loss])
    print(show_ssim, show_psnr, show_l1, show_l2)
