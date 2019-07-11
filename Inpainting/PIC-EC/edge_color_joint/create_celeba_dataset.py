import os
import argparse
import numpy as np
import cv2
from scipy.misc import imread
from scipy.misc import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the output dataset')
args = parser.parse_args()

ext = {'.jpg', '.png', '.JPG'}

i = 1
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            file_path = os.path.join(root, file)

            print('handle the {}st image.'.format(i))
            img = imread(file_path)
            a = img[20:20 + 178, :, :]
            a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
            imsave(os.path.join(args.output, '%06d.png' % i), a)

            i = i + 1


# img = imread('img.JPG')
# print(img.shape)
# a = img[:, 0:537, :]
# b = img[:, 199:736, :]
# c = img[:, 399:936, :]

# print(a.shape, b.shape, c.shape)

# a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
# b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)
# c = cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA)

# for i in range(20):
#     filename = 'psv%05d.JPG' % i
#     print(filename)

# path = 'F:\\Datasets\\psv\\train'

# imsave(os.path.join(path, 'a.JPG'), a)
# imsave(os.path.join(path, 'b.JPG'), b)
# imsave(os.path.join(path, 'c.JPG'), c)
