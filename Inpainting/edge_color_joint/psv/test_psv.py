import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

img = imread('img.JPG')
print(img.shape)
a = img[:, 0:537, :]
b = img[:, 199:736, :]
c = img[:, 399:936, :]

print(a.shape, b.shape, c.shape)

imsave('a.JPG', a)
imsave('b.JPG', b)
imsave('c.JPG', c)
