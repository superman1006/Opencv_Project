import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示

# 图像阈值
# ① ret, dst = cv2.threshold(src, thresh, maxval, type)
# - src： 输入图，只能输入单通道图像，通常来说为灰度图
# - thresh： 阈值
# - dst： 输出图
# - ret： 阈值
# - maxval： 当像素值超过了阈值 ( 或者小于阈值，根据 type 来决定 )，所赋予的值
# - type：二值化操作的类型，包含以下5种类型：
# -     cv2.THRESH_BINARY         超过阈值部分取maxval ( 最大值 )，否则取0
# -     cv2.THRESH_BINARY_INV     THRESH_BINARY的反转
# -     cv2.THRESH_TRUNC          大于阈值部分设为阈值，否则不变
# -     cv2.THRESH_TOZERO         大于阈值部分不改变，否则设为0
# -     cv2.THRESH_TOZERO_INV    THRESH_TOZERO的反转

img = cv2.imread('references/people.png')
img_gray = cv2.imread('references/people.png', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
print(ret)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# THRESH_BINARY_INV 相对 THRESH_BINARY 黑的变成白的，白的变成黑的
print(ret)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
print(ret)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
print(ret)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
print(ret)

titles = ['original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("results/图像阈值.png")
plt.show()
