import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from skimage.feature import canny
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from test_blur import get_color_domain

img = imread('img.png')
img_gray = rgb2gray(img)
edge = canny(img_gray, sigma=2)
img_color = get_color_domain(img, 21, 3, 3)

h, w, c = img.shape
hole = h // 2
# top = np.random.randint(0, hole + 1)
# left = np.random.randint(0, hole + 1)
top = hole // 2
left = hole // 2

mask = np.pad(array=np.ones((hole, hole)),
              pad_width=((top, h - hole - top), (left, w - hole - left)),
              mode='constant',
              constant_values=0)

img_masked = img * (1 - np.expand_dims(mask, -1)) + np.expand_dims(mask, -1) * 255
gray_masked = img_gray * (1 - mask) + mask
edge_masked = (1 - edge) * (1 - mask) + mask
img_color_masked = img_color * (1 - np.expand_dims(mask, -1)) + np.expand_dims(mask, -1)

imsave('gray_masked.png', gray_masked)
imsave('edge_masked.png', edge_masked)
imsave('mask.png', mask)
imsave('edge.png', 1 - edge)
imsave('img_masked.png', img_masked)
imsave('img_color.png', img_color)
imsave('img_color_masked.png', img_color_masked)

plt.figure()

plt.subplot(121)
plt.imshow(gray_masked, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img_masked')

plt.subplot(122)
plt.imshow(edge_masked, cmap=plt.cm.gray)
plt.axis('off')
plt.title('edge_masked')


plt.show()
