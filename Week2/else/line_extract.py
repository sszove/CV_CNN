import cv2
import numpy as np

img = cv2.imread("line.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# yellow and white color selection
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([35, 255, 255])

white = cv2.inRange(img, lower_white, upper_white)
yellow = cv2.inRange(img, lower_yellow, upper_yellow)

# 混合黄白车道线
mixed_img = cv2.bitwise_or(white, yellow)

# 使用ROI提取车道
row, col = mixed_img.shape[:2]
# 定义多边形的顶点
bottom_left = [col * 0.05, row]
top_left = [col * 0.45, row * 0.6]
top_right = [col * 0.55, row * 0.6]
bottom_right = [col * 0.95, row]
# 使用顶点定义多边形
vertices = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)

# 生成一个与原始图像同等大小的掩膜图像
roi_mask = np.zeros((row, col), dtype=np.uint8)
cv2.fillPoly(roi_mask, [vertices], 255)
# roi
roi_img = cv2.bitwise_and(mixed_img, mixed_img, mask=roi_mask)

# 高斯模糊，Canny边缘检测需要的
lane = cv2.GaussianBlur(roi_img, (5, 5), 0)
# 进行边缘检测，减少图像空间中需要检测的点数量
lane = cv2.Canny(lane, 50, 150)
cv2.imshow("lane", lane)
cv2.waitKey()

# hough 直线边缘提取
rho = 1  # 距离分辨率
theta = np.pi / 180  # 角度分辨率
threshold = 10  # 霍夫空间中多少个曲线相交才算作正式交点
min_line_len = 10  # 最少多少个像素点才构成一条直线
max_line_gap = 50  # 线段之间的最大间隔像素
lines = cv2.HoughLinesP(lane, rho, theta, threshold, maxLineGap=max_line_gap)
line_img = np.zeros_like(lane)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
cv2.imshow("line_img", line_img)
cv2.waitKey()
