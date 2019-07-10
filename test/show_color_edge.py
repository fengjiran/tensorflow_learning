import numpy as np
from imageio import imread
from imageio import imwrite
import matplotlib.pyplot as plt

# edge = imread('/Users/richard/Desktop/edge.png')  # (256, 256)
# mask = imread('/Users/richard/Desktop/mask.png')  # (256, 256)

edge = imread('E:\\model\\sample\\test_02.png')
mask = imread('E:\\model\\sample\\mask_02.png')

edge = edge / 255
mask = mask / 255

tmp = edge * mask

r = tmp * 0 / 255.
g = tmp * 0 / 255.
b = tmp * 255 / 255.

r = np.expand_dims(r, -1)
g = np.expand_dims(g, -1)
b = np.expand_dims(b, -1)

m = np.concatenate((r, g, b), axis=2)
# m = m.astype(np.uint8)

edge = np.expand_dims(edge, -1)  # (256, 256, 1)
# mask = np.expand_dims(mask, -1)  # (256, 256, 1)


edges = np.concatenate((edge, edge, edge), axis=2)  # (256, 256, 3)
# masks = np.concatenate((mask, mask, mask), axis=2)  # (256, 256, 3)

s = 1 - edges + m

# imwrite('/Users/richard/Desktop/tmp.png', s)
imwrite('E:\\model\\sample\\inpaint_edge_01.png', s)

plt.imshow(s)
plt.show()
