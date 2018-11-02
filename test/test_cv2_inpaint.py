import numpy as np
import cv2 as cv


def random_bbox(img_height, img_width, hole_height, hole_width):
    top = np.random.randint(low=0, high=img_height - hole_height)
    left = np.random.randint(low=0, high=img_width - hole_width)

    return (top, left, hole_height, hole_width)


def bbox2mask_np(bbox, height, width):
    top, left, h, w = bbox
    mask = np.pad(array=np.ones((h, w)),
                  pad_width=((top, height - h - top), (left, width - w - left)),
                  mode='constant',
                  constant_values=0)
    # mask = np.expand_dims(mask, 0)
    # mask = np.expand_dims(mask, -1)
    # mask = np.concatenate((mask, mask, mask), axis=3) * 255
    return mask


if __name__ == '__main__':
    pass
