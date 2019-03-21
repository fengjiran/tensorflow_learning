import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

img = imread('F:\\Datasets\\qd_imd\\train\\00001_train.png')
img1 = imresize(img, [256, 256])

plt.figure(figsize=(8, 3))

plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('origin', fontsize=20)

plt.subplot(122)
plt.imshow(img1)
plt.axis('off')
plt.title('resize', fontsize=20)
