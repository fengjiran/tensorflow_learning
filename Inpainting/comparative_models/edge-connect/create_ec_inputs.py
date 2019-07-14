import os
from imageio import imread
from imageio import imwrite
import numpy as np
import cv2

img_path = 'E:\\model\\experiments\\exp2\\celebahq\\gt_images\\256'
irregular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'
regular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\regular_mask'

irregular_masked_img_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_images\\irregular'
regular_masked_img_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_images\\regular'

irregular_mask_reverse_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_masks\\irregular'
regular_mask_reverse_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_masks\\regular'

img_dir = os.listdir(img_path)
irregular_mask_dir = os.listdir(irregular_mask_path)
regular_mask_dir = os.listdir(regular_mask_path)

# create regular masked input images
i = 1
for dir1, dir2 in zip(img_dir, regular_mask_dir):
    img = imread(os.path.join(img_path, dir1))  # (256, 256, 3)
    mask = imread(os.path.join(regular_mask_path, dir2))  # (256, 256)
    mask = np.expand_dims(mask, -1)

    img = img / 255.
    mask = mask / 255.

    masked_img = img * mask + 1 - mask

    imwrite(os.path.join(regular_masked_img_path, 'regular_masked_img_256_%03d.png' % i), masked_img)

    mask = np.reshape(mask, (256, 256))
    imwrite(os.path.join(regular_mask_reverse_path, 'regular_mask_%03d.png' % i), 1 - mask)

    i += 1


# create irregular masked input images
j = 1
for dir1, dir2 in zip(img_dir, irregular_mask_dir):
    img = imread(os.path.join(img_path, dir1))  # (256, 256, 3)
    mask = imread(os.path.join(irregular_mask_path, dir2))  # (256, 256)
    mask = np.expand_dims(mask, -1)

    img = img / 255.
    mask = mask / 255.

    masked_img = img * mask + 1 - mask

    imwrite(os.path.join(irregular_masked_img_path, 'irregular_masked_img_256_%03d.png' % j), masked_img)

    mask = np.reshape(mask, (256, 256))
    imwrite(os.path.join(irregular_mask_reverse_path, 'irregular_mask_%03d.png' % j), 1 - mask)

    j += 1
