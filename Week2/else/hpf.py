# /usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread('lenna.jpg', 0)

# 高通滤波
kernel_3x3 = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                      [-1, 1, 2, 1, -1],
                      [-1, 2, 4, 2, -1],
                      [-1, 1, 2, 1, -1],
                      [-1, -1, -1 , -1, -1]])

k3 = ndimage.convolve(img, kernel_3x3)

k5 = ndimage.convolve(img, kernel_5x5)

# Gaussian Kernel Effect
g_img = cv2.GaussianBlur(img,(11,11),0)
# 高通滤波，突出边缘
g_hpf = img - g_img

cv2.imshow('k3_lenna', k3)
cv2.imshow('k5_lenna', k5)
cv2.imshow('gaussian_blur_lenna', g_img)
cv2.imshow('hpf_lenna', g_hpf)
key = cv2.waitKey()
cv2.destroyAllWindows()