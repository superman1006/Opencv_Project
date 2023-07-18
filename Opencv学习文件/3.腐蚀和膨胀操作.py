import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------腐蚀操作-------------------------------------------------

img1 = cv2.imread('references/DiGe.png')
cv2.imshow('img', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8)  # 用numpy创建一个5*5的卷积核
erosion_result1 = cv2.erode(img1, kernel, iterations=1)  # iterations迭代次数为1
erosion_result2 = cv2.erode(img1, kernel, iterations=2)  # iterations迭代次数为2

cv2.imshow('erosion1', erosion_result1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('erosion2', erosion_result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------------------------膨胀操作-------------------------------------------------


img2 = cv2.imread('references/DiGe.png')
cv2.imshow('img', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 先腐蚀 后膨胀，抵消腐蚀造成的损害
kernel = np.ones((3, 3), np.uint8)
dige_erosion = cv2.erode(img2, kernel, iterations=1)
cv2.imshow('erosion', dige_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3, 3), np.uint8)
dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
cv2.imshow('dilate', dige_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
