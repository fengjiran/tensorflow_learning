from __future__ import print_function
import os
# import sys
import glob
import numpy as np
import PIL.Image
import tensorflow as tf


def error(msg):
    print('Error: ' + msg)
    exit(1)


def create_celebahq_tfrecord(tfrecord_dir, celebahq_dir):
    num = 1000
    print('Loading CelebA from "%s"' % celebahq_dir)
    glob_pattern = os.path.join(celebahq_dir, '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    # print(len(image_filenames))
    # expected_images = 202599
    # expected_images = 182637
    expected_images = 1000
    # if len(image_filenames) != expected_images:
    #     error('Expected to find %d images' % expected_images)

    # num_tfrecords = expected_images // num + 1 if expected_images % num else expected_images // num
    # tfrecord_num = 1
    # tfrecord_name = 'celeba_traindata.tfrecord-%.3d' % tfrecord_num
    num_tfrecords = expected_images // num
    order = np.arange(expected_images) + 29000
    # np.random.RandomState(123).shuffle(order)

    cur_img = 1
    # for i in range(num_tfrecords - 1):
    #     print('write ', i + 1, ' file')
    #     tfrecordname = os.path.join(tfrecord_dir, 'celebahq_trainset.tfrecord-%.3d' % (i + 1))
    #     writer = tf.python_io.TFRecordWriter(path=tfrecordname)
    #     for j in range(num):
    #         print('write ', cur_img, ' image')
    #         img = np.asarray(PIL.Image.open(image_filenames[order[i * num + j]]))
    #         assert img.shape == (1024, 1024, 3)
    #         # img = img.astype(np.float32)
    #         # img = img / 127.5 - 1
    #         example = tf.train.Example(
    #             features=tf.train.Features(
    #                 feature={
    #                     'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
    #                     'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
    #                 }
    #             ))
    #         writer.write(record=example.SerializeToString())
    #         cur_img += 1
    #     writer.close()

    print('write ', num_tfrecords, ' file')
    tfrecordname = os.path.join(tfrecord_dir, 'celebahq_valset.tfrecord-%.3d' % num_tfrecords)
    writer = tf.python_io.TFRecordWriter(path=tfrecordname)
    for idx in order[(num_tfrecords - 1) * num:]:
        print('write ', cur_img, ' image')
        img = np.asarray(PIL.Image.open(image_filenames[idx]))
        assert img.shape == (1024, 1024, 3)
        # img = img.astype(np.float32)
        # img = img / 127.5 - 1
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
                }
            ))
        writer.write(record=example.SerializeToString())
        cur_img += 1
    writer.close()


if __name__ == '__main__':
    celebahq_dir = 'F:\\Datasets\\celebahq'
    tfrecord_dir = 'F:\\Datasets\\celebahq_tfrecords\\val'
    create_celebahq_tfrecord(tfrecord_dir, celebahq_dir)
