import os
import shutil
from glob import glob
import skimage.io
import skimage.transform
import skimage.measure
from PIL import Image

import numpy as np


def array_to_image(array):
    r = Image.fromarray(array[0]).convert('L')
    g = Image.fromarray(array[1]).convert('L')
    b = Image.fromarray(array[2]).convert('L')

    image = Image.merge('RGB', (r, g, b))

    return image


def load_image(path, pre_height=146, pre_width=146, height=128, width=128):
    try:
        # print path
        img = skimage.io.imread(path).astype(float)
        if img.shape[0] < pre_height or img.shape[1] < pre_width:
            pass
            # print path
            # print img.shape
    except TypeError:
        return None

    img /= 255.

    if img is None:
        return None

    # The shape of image: (height, width, channel)
    if len(img.shape) < 2:
        return None

    if len(img.shape) == 4:
        return None

    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)

    if img.shape[2] == 4:
        img = img[:, :, 0:3]

    if img.shape[2] > 4:
        return None

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, [pre_height, pre_width])

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[rand_y:rand_y + height, rand_x:rand_x + width, :]
    # resized_img = np.transpose(resized_img, [2, 0, 1])  # convert to channel first

    return resized_img * 2 - 1  # [-1, 1] [128, 128, 3]


def crop_random(image_ori, width=64, height=64, x=None, y=None, overlap=7):

    if image_ori is None:
        return None

    random_y = np.random.randint(overlap, height - overlap) if x is None else x
    random_x = np.random.randint(overlap, width - overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()

    crop = crop[random_y:random_y + height, random_x:random_x + width, :]  # ground truth
    image[random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap,
          0] = 2 * 117. / 255. - 1.

    image[random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap,
          1] = 2 * 104. / 255. - 1.

    image[random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap,
          2] = 2 * 123. / 255. - 1.

    return image, crop, random_x, random_y
