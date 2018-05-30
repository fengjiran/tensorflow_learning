from __future__ import print_function
import os
import sys
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
    print('Loading CelebA from "%s"' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, 'img_align_celeba_png', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 202599
    if len(image_filenames) != expected_images:
        error('Expected to find %d images' % expected_images)

    order = np.arange(expected_images)
    np.random.RandomState(123).shuffle(order)

    for idx in range(order.size):
        print('%d / %d\r' % (idx + 1, expected_images), end='', flush=True)
        # img = PIL.Image.open(image_filenames[order[idx]])
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
        assert img.shape == (218, 178, 3)

    # with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
    #     order = tfr.choose_shuffled_order()
    #     for idx in range(order.size):
    #         img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
    #         assert img.shape == (218, 178, 3)
    #         img = img[cy - 64: cy + 64, cx - 64: cx + 64]
    #         img = img.transpose(2, 0, 1)  # HWC => CHW
    #         tfr.add_image(img)
