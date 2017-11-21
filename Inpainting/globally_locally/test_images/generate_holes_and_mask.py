import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

test1 = io.imread('1_origin.png').astype(float)
test2 = io.imread('1_with_holes.png').astype(float)

if test1.shape[2] == 4:
    test1 = test1[:, :, 0:3]

if test2.shape[2] == 4:
    test2 = test2[:, :, 0:3]

mask = np.abs(test1 - test2)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j, 0] + mask[i, j, 1] + mask[i, j, 2] != 0:
            mask[i, j, 0] = 1.
            mask[i, j, 1] = 1.
            mask[i, j, 2] = 1.

plt.subplot(131)
plt.imshow(test1.astype('uint8'))

plt.subplot(132)
plt.imshow(test2.astype('uint8'))

plt.subplot(133)
plt.imshow(((1 - mask) * 255.).astype('uint8'))

plt.show()
