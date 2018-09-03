import os
import sys
import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import tarfile
import zipfile
import scipy.io


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.)
            )
            sys.stdout.flush()


def get_model_data(dir_path, model_url):
    pass
