import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

img = imread('F:\\Datasets\\qd_imd\\train\\00002_train.png')
img1 = imresize(img, [256, 256])
img1 = (img1 > 3).astype(np.uint8) * 255

plt.figure(figsize=(8, 3))

plt.subplot(121)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.title('origin', fontsize=20)

plt.subplot(122)
plt.imshow(img1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('resize', fontsize=20)

plt.show()
