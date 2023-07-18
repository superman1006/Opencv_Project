import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包

# 礼帽 -- tophat
# tophat操作作用是将图片中的亮度高于周围的区域都突出出来！！！！！！！！！！！！！！！！
# 原始带刺，开运算不带刺，原始输入-开运算 = 刺
img = cv2.imread('references/DiGe.png')
kernel = np.ones((5, 5), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 黑帽 -- blackhat
# black操作作用是将图片中的亮度低于周围的区域都突出出来！！！！！！！！！！！！！！！！
# 原始带刺，闭运算带刺并且比原始边界胖一点，闭运算-原始输入 = 原始整体
img = cv2.imread('references/DiGe.png')
kernel = np.ones((5, 5), np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
