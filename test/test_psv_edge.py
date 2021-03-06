import numpy as np
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2

path = 'F:\\Datasets\\psv\\psv00000.JPG'
img = imread(path)
gray = rgb2gray(img)
edge = canny(gray, 1.8)

plt.figure(figsize=(8, 3))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title('image')

plt.subplot(132)
plt.imshow(gray, cmap=plt.cm.gray)
plt.axis('off')
plt.title('gray')

plt.subplot(133)
plt.imshow(edge, cmap=plt.cm.gray)
plt.axis('off')
plt.title('edge')

plt.show()
