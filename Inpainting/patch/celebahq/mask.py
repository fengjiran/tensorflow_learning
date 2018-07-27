import os
import cv2
import numpy as np


def bbox2mask_np(bbox, height, width):
    top, left, h, w = bbox
    mask = np.pad(array=np.ones((h, w)),
                  pad_width=((top, height - h - top), (left, width - w - left)),
                  mode='constant',
                  constant_values=0)
    # mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, -1)
    mask = np.concatenate((mask, mask, mask), axis=2)
    return mask


# img_path = 'F:\\Datasets\\celebahq\\img00029978.png'
img_path = 'F:\\Datasets\\celebahq\\img00029978.png'
hole_size = 30
image_size = 256
bbox_np = ((image_size - hole_size) // 2,
           (image_size - hole_size) // 2,
           hole_size,
           hole_size)
mask = bbox2mask_np(bbox_np, image_size, image_size)
image = cv2.imread(img_path)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

image = image * (1 - mask) + mask * 0.5 * 255
image = image.astype(np.uint8)
cv2.imwrite('F:\\mask_100.png', image)
