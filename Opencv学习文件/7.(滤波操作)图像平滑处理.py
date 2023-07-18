import cv2  # opencv的缩写为cv2
import numpy as np  # numpy数值计算工具包


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


img = cv2.imread('references/people_noise.png')
cv_show(img, 'img')
# 均值滤波
# 简单的的平均卷积操作
blur = cv2.blur(img, (3, 3))
cv_show(blur, 'blur')

# 方框滤波
# 基本和均值一样，可以选择归一化,容易越界
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# 这里的 -1 表示输出图像的深度与原图像保持一致
# 这里的normalize=True表示方框滤波器是归一化的，这样输出图像更加自然
cv_show(box, 'box')

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的值，越远离中心点权重越小
gaussian = cv2.GaussianBlur(img, (5, 5), 1)
cv_show(gaussian, 'gaussian')

# 中值滤波
# 相当于用中值代替,适合椒盐噪声
median = cv2.medianBlur(img, 5)
cv_show(median, 'median')

# 展示所有的图像
res = np.hstack((blur, box, gaussian, median))
cv_show(res, 'result')

# 这几种当中，高斯滤波应用最为广泛，中值滤波也常用于去除椒盐噪声
