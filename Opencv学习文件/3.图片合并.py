import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图片大小调整resize
img1 = cv2.imread('references/Desktop1.png')
img2 = cv2.imread('references/Desktop2.png')
print(img1.shape)  # (998, 1230, 3)
print(img2.shape)  # (568, 423, 3)
img1 = cv2.resize(img1, (423, 568))  # 调整图片大小让img1的shape与img2一致

img_test = cv2.resize(img1, (0, 0), fx=2, fy=2)  # fx,fy为缩放因子,将图片的x,y方向分别放大2倍
cv2.imshow('img_test', img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图片合并addWeighted
result = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)  # 图片融合
# addWeighted()函数的参数为两张图片，两张图片的权重，以及gamma值
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('results/merge_test1_test2.png', result)  # 保存图片到本地






