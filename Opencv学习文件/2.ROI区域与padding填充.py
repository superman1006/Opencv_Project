import cv2
import numpy as np
import matplotlib.pyplot as plt

#  ROI区域 reign of interest

img = cv2.imread("references/Desktop1.png")

img_roi = img[100:300, 100:400]  # ROI区域,高度从50~100，宽度从50~100
cv2.imshow("ROI", img_roi)
cv2.waitKey(0)  # 等待时间，毫秒级，0表示任意键终止
cv2.destroyAllWindows()

b, g, r = cv2.split(img)  # 通道分离,将BGR三个通道分离出来，得到三个二维矩阵
print(b)
print(b.shape)
print(g)
print(g.shape)
print(r)
print(r.shape)

img = cv2.merge((b, g, r))  # 通道合并，将三个BGR通道合并成一个图像
print(img.shape)

# padding边界填充

top_size, bottom_size, left_size, right_size = (100, 100, 100, 100)  # 上下左右填充的像素数

replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                              value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.savefig("results/padding.png")  # 用plt保存图片
plt.show()

# plt.subplot(231)中的数字代表的是：plt.subplot(2,2,1)中的2代表的是将整个图像窗口分为2行，3列，1代表的是图像画在从左到右从上到下的第1块
# BORDER_REPLICATE：复制法，也就是复制最边缘像素。
# BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
# BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
# BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
# BORDER_CONSTANT：常量法，常数值填充。
