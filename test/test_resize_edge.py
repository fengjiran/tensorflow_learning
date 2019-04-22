import numpy as np
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2


def get_color_domain(img, blur_factor1, blur_factor2, k):  # img:[0, 255], uint8
    img_blur = cv2.medianBlur(img, blur_factor1)
    Z = img_blur.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 8
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img_blur.shape))

    img_color_domain = cv2.medianBlur(res, blur_factor2)

    img_color_domain = img_color_domain / 255.
    img_color_domain = img_color_domain.astype(np.float32)
    return img_color_domain  # [0, 1]


if __name__ == '__main__':
    path = 'F:\\Datasets\\celebahq\\img00000001.png'
    image = imread(path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    color = get_color_domain(image, 21, 3, 3)
    gray = rgb2gray(image)
    edge = canny(gray, 2)
    edge = edge.astype(np.float32)

    edge1 = cv2.resize(edge, (128, 128), interpolation=cv2.INTER_AREA)
    edge1 = edge1 > 0.25

    color1 = cv2.resize(color, (128, 128), interpolation=cv2.INTER_AREA)

    # plt.figure(figsize=(8, 3))

    plt.subplot(161)
    plt.imshow(image)
    plt.axis('off')
    plt.title('image')

    plt.subplot(162)
    plt.imshow(gray, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('gray')

    plt.subplot(163)
    plt.imshow(edge, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('edge')

    plt.subplot(164)
    plt.imshow(edge1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('edge')

    plt.subplot(165)
    plt.imshow(color)
    plt.axis('off')
    plt.title('color')

    plt.subplot(166)
    plt.imshow(color1)
    plt.axis('off')
    plt.title('color')

    plt.show()
