import os
from imageio import imread
from imageio import imwrite
import cv2

img = imread('ref.png')
mask = imread('irregular_mask_001.png')
mask = 255 - mask

dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
imwrite('./dst.png', dst)
