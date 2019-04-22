import numpy as np
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import cv2

path = 'F:\\Datasets\\celebahq\\img00000001.png'
image = imread(path)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
gray = rgb2gray(image)
edge = canny(gray, 2)
edge = edge.astype(np.float32)
