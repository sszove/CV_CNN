# -*- coding: utf-8 -*-

"""
Time: 2019.07.24
Author: ssz
Function: Median Blur with difference padding
Thought: original image (x,y) map to padding image (x + padding, y + padding)
Key: padding -0 or near
Ref: https://blog.csdn.net/kingroc/article/details/96166953
"""

import cv2
import numpy as np


def myMedianBlur(img, kernel, padding_way='ZERO'):
    # 检测传入的kernel的大小：大于3的基数
    if kernel % 2 == 0 or kernel is 1:
        print('kernel size need 3, 5, 7, 9....')
        return None

    # 通过kernel的大小来计算paddingSize的大小
    padding_size = kernel // 2

    # 获取图片的通道数
    layer_size = len(img.shape)

    # 获取传入的图片的大小
    height, width = img.shape[:2]

    # 多通道递归单通道
    if layer_size == 3:
        mut_ch_img = np.zeros_like(img)
        for l in range(mut_ch_img.shape[2]):
            mut_ch_img[:, :, l] = myMedianBlur(img[:, :, l], kernel, padding_way)
        return mut_ch_img
    # 单通道处理
    elif layer_size == 2:
        # 实现方式和np.lib.pad相同
        # matBase = np.lib.pad(img,paddingSize, mode='constant', constant_values=0)
        img_padding = np.zeros((height + padding_size * 2, width + padding_size * 2), dtype=img.dtype)
        img_padding[padding_size:-padding_size, padding_size:-padding_size] = img
        # 将原值写入新创建的矩阵当中
        if padding_way is 'ZERO':
            # padding 0  无变化
            pass
        elif padding_way is 'REPLICA':
            # REPLICA, 四个边补齐,填充临近值
            for r in range(padding_size):
                # top
                img_padding[r, padding_size:-padding_size] = img[0, :]
                # bottom
                img_padding[-(1 + r), padding_size:-padding_size] = img[-1, :]
                # left
                img_padding[padding_size:-padding_size, r] = img[:, 0]
                # right
                img_padding[padding_size:-padding_size, -(1 + r)] = img[:, -1]
        else:
            print('padding_way error need ZERO or REPLICA')
            return None

        # 创建用于输出的矩阵
        img_out = np.zeros((height, width), dtype=img.dtype)
        # 这里是遍历原图的每个坐标
        for x in range(height):
            for y in range(width):
                # kernel * kernel 的矩阵转化成list
                line = img_padding[x:x + kernel, y:y + kernel].flatten()
                line = np.sort(line)
                # 取中间值赋值
                img_out[x, y] = line[(kernel * kernel) // 2]
        return img_out
    else:
        print('image layers error')
        return None


'''
椒盐噪声也叫脉冲噪声，即在一幅图像里随机将一个像素点变为椒噪声或盐噪声，其中椒噪声像素值为“0”，盐噪声像素值为“255”。
生成（添加）椒盐噪声算法步骤如下：
（1）输入一幅图像并自定义信噪比 SNR （其取值范围在[0, 1]之间）；
（2）计算图像像素点个数 SP， 进而得到椒盐噪声的像素点数目 NP = SP * (1-SNR)；
（3）随机获取要加噪的每个像素位置img[i, j]；
（4）随机生成[0,1]之间的一个浮点数；
（5）判定浮点数是否大于0.5，并指定像素值为255或者0；
（6）重复3，4，5三个步骤完成所有像素的NP个像素加粗样式；
（7）输出加噪以后的图像。
'''


def rgb2gray(img):
    h = img.shape[0]
    w = img.shape[1]
    img_out = np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            img_out[i, j] = 0.144 * img[i, j, 0]+0.587*img[i, j, 1]+0.299*img[i, j, 2]
    return img_out


def noise(img, snr):
    h = img.shape[0]
    w = img.shape[1]
    img_out = img.copy()
    sp = h * w   # 计算图像像素点个数
    NP = int(sp * (1 - snr))   # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img_out[randx, randy]=0
        else:
            img_out[randx, randy]=255
    return img_out


def main():
    # 读取原始图片
    img = cv2.imread('naroto.jpg')
    img = img[200:500, 200:500]

    gray_image = rgb2gray(img)
    img = noise(gray_image, 0.6)  # 将信噪比设定为0.6

    # 手写的medianBlur
    img_rep = myMedianBlur(img, 5, padding_way='REPLICA')
    if img_rep is None:
        return

    img_zero = myMedianBlur(img, 5, padding_way='ZERO')
    if img_zero is None:
        return

    # 调用OpenCV的接口进行中值滤波
    cv_img = cv2.medianBlur(img, 5)

    # 这里进行图片合并
    img1 = np.hstack((img_rep, img))
    img2 = np.hstack((img_zero, cv_img))
    img3 = np.vstack((img1, img2))

    # 显示对比效果
    cv2.imshow('padding rep-->origin-->padding zero-->cv', img3)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
