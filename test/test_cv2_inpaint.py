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
    mask = np.expand_dims(mask, -1)
    # mask = np.concatenate((mask, mask, mask), axis=3) * 255
    return mask


if __name__ == '__main__':
    img = cv.imread('val.png')
    height = img.shape[0]
    width = img.shape[1]

    bbox = random_bbox(height, width, 128, 128)
    mask = bbox2mask_np(bbox, height, width)

    img = img / 127.5 - 1

    destroy = (img + 1.) * 0.5 * (1. - mask)
    destroy = destroy * 255.
    destroy = destroy.astype(np.uint8)

    # src = img * (1.0 - mask) * 255.
    dst = cv.inpaint(destroy, mask.astype(np.uint8), 5, cv.INPAINT_TELEA)
    dst = dst.astype(np.uint8)
    print(dst[0][0][2])

    cv.imshow('img', (img + 1.) * 0.5)
    cv.imshow('src', destroy)
    cv.imshow('dst', dst)
    cv.waitKey(0)

    # dst = cv.inpaint()
