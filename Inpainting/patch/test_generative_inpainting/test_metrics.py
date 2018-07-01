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

    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)
    mask = tf.concat([mask, mask, mask], axis=3)
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

val_filenames = tf.placeholder(tf.string, shape=[None])
val_data = tf.data.TFRecordDataset(val_filenames)
val_data = val_data.map(parse_tfrecord)
val_data = val_data.batch(10)
# val_data = val_data.repeat()
val_iterator = val_data.make_initializable_iterator()
val_batch_data = val_iterator.get_next()
val_batch_data = tf.image.resize_area(val_batch_data, [256, 256])
val_batch_data = tf.clip_by_value(val_batch_data, 0., 255.)
val_batch_data = val_batch_data / 127.5 - 1
# val_batch_data = tf.placeholder(tf.float32, shape=[10,256,256,3])
hole_size = 16
image_size = 256
bbox = (tf.constant((image_size - hole_size) // 2),
        tf.constant((image_size - hole_size) // 2),
        tf.constant(hole_size),
        tf.constant(hole_size))

mask = bbox2mask(bbox, image_size, image_size)
ones_x = tf.ones_like(val_batch_data)
mask = ones_x * mask
input_image = tf.concat([val_batch_data, mask], axis=2)
# print(mask.get_shape())
print(input_image.get_shape())
# ng.get_gpus(1)
# args = parser.parse_args()
model = InpaintCAModel()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    # input_image = tf.constant(input_image, dtype=tf.float32)
    output = model.build_server_graph(input_image)