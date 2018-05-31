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


class TFRecordExporter(object):
    """Create tfrecord dataset."""

    def __init__(self, tfrecord_dir, expected_images,
                 print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.expected_images = expected_images
        self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.cur_images = 0
        self.shape = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order


def create_celeba_tfrecord(tfrecord_dir, celeba_dir):
    num = 1000
    print('Loading CelebA from "%s"' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    # print(len(image_filenames))
    # expected_images = 202599
    # expected_images = 182637
    expected_images = 9981
    # if len(image_filenames) != expected_images:
    #     error('Expected to find %d images' % expected_images)

    # num_tfrecords = expected_images // num + 1 if expected_images % num else expected_images // num
    # tfrecord_num = 1
    # tfrecord_name = 'celeba_traindata.tfrecord-%.3d' % tfrecord_num
    num_tfrecords = expected_images // num
    order = np.arange(expected_images) + 182637 + 9981
    # np.random.RandomState(123).shuffle(order)

    cur_img = 1
    for i in range(num_tfrecords - 1):
        print('write ', i + 1, ' file')
        tfrecordname = os.path.join(tfrecord_dir, 'celeba_testset.tfrecord-%.3d' % (i + 1))
        writer = tf.python_io.TFRecordWriter(path=tfrecordname)
        for j in range(num):
            print('write ', cur_img, ' image')
            img = np.asarray(PIL.Image.open(image_filenames[order[i * num + j]]))
            assert img.shape == (218, 178, 3)
            img = img.astype(np.float32)
            img = img / 127.5 - 1
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

    print('write ', num_tfrecords, ' file')
    tfrecordname = os.path.join(tfrecord_dir, 'celeba_testset.tfrecord-%.3d' % num_tfrecords)
    writer = tf.python_io.TFRecordWriter(path=tfrecordname)
    for idx in order[(num_tfrecords - 1) * num:]:
        print('write ', cur_img, ' image')
        img = np.asarray(PIL.Image.open(image_filenames[idx]))
        assert img.shape == (218, 178, 3)
        img = img.astype(np.float32)
        img = img / 127.5 - 1
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
    celeba_dir = 'F:\\Datasets\\CelebA\\Img\\img_align_celeba_png.7z\\img_align_celeba_png'
    tfrecord_dir = 'F:\\Datasets\\celeba_tfrecords\\test'
    create_celeba_tfrecord(tfrecord_dir, celeba_dir)
