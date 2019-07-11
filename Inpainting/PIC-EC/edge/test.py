import os
import glob
import yaml
import platform as pf
import numpy as np
import cv2
from imageio import imread
from imageio import imwrite
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray

image_paths = ['C:\\Users\\Richard\\Desktop\\img00000020.png', 'C:\\Users\\Richard\\Desktop\\img00000590.png',
               'C:\\Users\\Richard\\Desktop\\img00000934.png', 'C:\\Users\\Richard\\Desktop\\img00001469.png']
mask_paths = ['C:\\Users\\Richard\\Desktop\\00127_test.png', 'C:\\Users\\Richard\\Desktop\\00007_test.png',
              'C:\\Users\\Richard\\Desktop\\00034_test.png', 'C:\\Users\\Richard\\Desktop\\00052_test.png']


sample_dir = "E:\\model\\sample"

# i = 0
# for path in image_paths:
#     i = i + 1
#     img = imread(path)
#     img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
#     imwrite(os.path.join(sample_dir, 'img_%02d.png' % i), img)

i = 0
for path in mask_paths:
    i = i + 1
    mask = imread(path)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask = mask > 3
    mask = mask.astype(np.float32)
    imwrite(os.path.join(sample_dir, 'mask_%02d.png' % i), 1 - mask)


# i = 0
# for img_path in image_paths:
#     for mask_path in mask_paths:
#         i = i + 1
#         img = imread(img_path)
#         mask = imread(mask_path)

#         img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # (256, 256, 3)
#         img = img / 255.

#         mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
#         mask = mask > 3
#         mask = mask.astype(np.float32)

#         mask = np.expand_dims(mask, -1)
#         mask = np.concatenate((mask, mask, mask), axis=2)

#         masked_img = img * mask + 1 - mask

#         imwrite(os.path.join(sample_dir, 'masked_img_%02d.png' % i), masked_img)
