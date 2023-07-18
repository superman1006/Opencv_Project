import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包


# -------------------------------Canny边缘检测-------------------------------
# 1.高斯模糊
# Canny边缘检测流程:
#   - 1)  使用高斯滤波器，以平滑图像，滤除噪声。
#   - 2)  计算图像中每个像素点的梯度强度和方向。
#   - 3)  应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
#   - 4)  应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
#   - 5)  通过抑制孤立的弱边缘最终完成边缘检测。

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 2.高斯滤波器
# 高斯滤波器靠近的中心点的权重比较大，较远中心点的权重比较小。
cv_show(cv2.imread('references/Canny1.png'), '高斯滤波器')
# 3.梯度和方向
cv_show(cv2.imread('references/Canny2.png'), '梯度和方向')

# 4.非极大值抑制
cv_show(cv2.imread('references/Canny3.png'), '非极大值抑制1')
cv_show(cv2.imread('references/Canny4.png'), '非极大值抑制2')

# 5.双阈值检测
cv_show(cv2.imread('references/Canny5.png'), '双阈值检测')

# 6.Canny代码实现
img = cv2.imread('references/people.png', cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 80, 150)
# 第二个参数为minVal，第三个参数为maxVal, 低于minVal的像素点会被抛弃，高于maxVal的像素点会被保留
v2 = cv2.Canny(img, 50, 100)
# minVal降低会导致检测到更多的边缘，但是也会增加噪声的影响
# maxVal提高会使得边缘检测的结果更加精准，但是也会导致边缘丢失

res = np.hstack((v1, v2))  # 水平拼接
cv_show(res, 'res')
