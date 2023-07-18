import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包

# 1.开：先腐蚀，再膨胀
img = cv2.imread('references/DiGe.png')

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.闭：先膨胀，再腐蚀
img = cv2.imread('references/DiGe.png')

kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.梯度 = 腐蚀 减去 膨胀 (常用于得到图像的边缘信息)
pie = cv2.imread('references/pie.png')

kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)

res = np.hstack((dilate, erosion))

cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
