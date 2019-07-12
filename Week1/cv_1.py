# /usr/bin/python
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# 显示灰度图片，并打印图片的灰度矩阵、矩阵类型、图片大小
def img_grey_test(img_gray):
    cv2.imshow('lenna', img_gray)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    # to show gray image to show image matrix
    print(img_gray)
    # to show image data type
    print(img_gray.dtype)
    # to show gray image shape return tuple (height, width)
    print(img_gray.shape)


# 显示正常的图片
def img_normal_test(img):
    cv2.imshow('lenna', img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    # to show color image to show channels
    print(img)
    # h, w, channel
    print(img.shape)


# 图像裁剪
def image_crop(img):
    img_crop = img[0:100, 0:200]
    cv2.imshow('img_crop', img_crop)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


# 图像通道值获取
def image_show_per_channel(img):
    B, G, R = cv2.split(img)
    cv2.imshow('B', B)
    cv2.imshow('G', G)
    cv2.imshow('R', R)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


# 随机改变图像各通道的像素值
def random_light_color(img):
    # 三通道值
    b, g, r = cv2.split(img)
    for channel in (b, g, r):
        rand_value = random.randint(-50, 50)
        if rand_value == 0:
            pass
        elif rand_value > 0:
            # 保证通道值在[0,255]
            lim = 255 - rand_value
            channel[channel > lim] = 255
            channel[channel <= lim] = (rand_value + channel[channel <= lim]).astype(img.dtype)
        elif rand_value < 0:
            lim = 0 - rand_value
            channel[channel < lim] = 0
            channel[channel >= lim] = (rand_value + channel[channel >= lim]).astype(img.dtype)
    img_merge = cv2.merge((b, g, r))
    return img_merge


# gamma 变换
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


# histogram 直方图
def img_show_histogram(img_brighter):
    img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
    plt.hist(img_brighter.flatten(), 256, [0, 256], color='r')
    img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])   # only for 1 channel
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(灰阶值), u&v: 色度
    cv2.imshow('Color input image', img_small_brighter)
    cv2.imshow('Histogram equalized', img_output)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


# rotation
def img_rotation_show(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)  # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna', img_rotate)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

    print(M)

    M[0][2] = M[1][2] = 0
    print(M)
    img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow('rotated lenna2', img_rotate2)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


# scale + rotation + translation = similarity transform 仿射变换
def img_show_similarity_transform(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 0.5) # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # Affine Transform
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('affine lenna', dst)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


# perspective transform 透视变换
def random_warp(img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


def main():
    # 默认为BGR，plt画图是GRB
    img_bgr = cv2.imread('/Users/ssz/Work/MachineLearning/CV_Lessons/Week1/lenna.jpg')
    img_gray = cv2.imread('/Users/ssz/Work/MachineLearning/CV_Lessons/Week1/lenna.jpg', 0)

    img_grey_test(img_gray)

    img_normal_test(img_bgr)

    image_crop(img_bgr)

    image_show_per_channel(img_bgr)

    img_random_color = random_light_color(img_bgr)
    cv2.imshow('img_random_color', img_random_color)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

    img_brighter = adjust_gamma(img_bgr, 2)

    img_show_histogram(img_brighter)

    img_rotation_show(img_bgr)

    img_show_similarity_transform(img_bgr)

    random_warp(img_bgr)

    M_warp, img_warp = random_warp(img_bgr)
    cv2.imshow('lenna_warp', img_warp)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
