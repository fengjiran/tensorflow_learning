import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2


# def img_kmeans(img_blur, K=8):
#     Z = img_blur.reshape((-1, 3))

#     # convert to np.float32
#     Z = np.float32(Z)

#     # define criteria, number of clusters(K) and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     # K = 8
#     ret, label, center = cv2.kmeans(Z, K, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res = res.reshape((img_blur.shape))
#     return res


def get_color_domain(img, blur_factor1, blur_factor2, k):
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
    return img_color_domain


if __name__ == '__main__':
    img = imread('img.png')
    img = img
    print(img.shape)

    img_color_domain = get_color_domain(img, 11, 3, 3)

    # img_color_domain = cv2.medianBlur(img, 11)
    # img_color_domain = img_kmeans(img_color_domain, k)
    # img_color_domain = cv2.medianBlur(img_color_domain, 3)

    print(img_color_domain.shape)

    plt.figure()

    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.title('rgb', fontsize=20)

    plt.subplot(122)
    plt.imshow(img_color_domain)
    plt.axis('off')
    plt.title('color_domain', fontsize=20)

    plt.show()
