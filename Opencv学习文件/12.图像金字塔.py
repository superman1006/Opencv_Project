import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 图像金字塔

# ------------------------------------------------- 1. 图像金字塔简介 ------------------------------------------------------


# ① 金字塔的底层是比较大，越往上越小，图像金字塔就是把图像组合成金字塔的形状。
# ② 图像金字塔可以做图像特征提取，做特征提取时有时可能不光对原始输入做特征提取，可能还会对好几层图像金字塔做特征提取。可能每一层特征提取的结果是不一样的，再把特征提取的结果总结在一起。
# ③ 常用的 两种 图像金字塔形式：
#       ( 用于图像向下采样时 , 将所有的偶数行和列去除 )
#       ( 用于图像向上采样时 , 将偶数行和列用 0 填充 )
#       - 1. 高斯金字塔
#       - 2. 拉普拉斯金字塔


# -------------------------------------------------- 2.高斯金字塔 ------------------------------------------------------

# 2.1 向下采样方法 ( 缩小 )

# 2.2 向上采样方法 ( 放大 )

img = cv2.imread('references/AM.png')  # 读取图片
cv_show(img, 'img')
print(img.shape)

# 上采样 1次
up_once = cv2.pyrUp(img)
cv_show(up_once, 'up_once')
print(up_once.shape)

# 下采样 1次
down_once = cv2.pyrDown(img)
cv_show(down_once, 'down_once')
print(down_once.shape)

# 上采样  之后再  上采样
up_twice = cv2.pyrUp(up_once)
cv_show(up_twice, 'up_twice')
print(up_twice.shape)

# 上采样  之后再  下采样
up_add_down = cv2.pyrDown(up_once)
cv_show(np.hstack((img, up_add_down)), 'img & up_add_down')
# 对比发现up_add_down 与 img 有一定的差异，这是因为在向下采样的时候，会丢失一些信息，所以再向上采样的时候，就会出现一些差异。

# -------------------------------------------------- 3. 拉普拉斯金字塔--------------------------------------------------

img = cv2.imread('references/AM.png')  # 读取图片

# 原图img 通过 下采样  之后再  上采样  , 再与原图相减
down = cv2.pyrDown(img)
down_add_up = cv2.pyrUp(down)
result = img - down_add_up
cv_show(result, 'result')
print(result.shape)
