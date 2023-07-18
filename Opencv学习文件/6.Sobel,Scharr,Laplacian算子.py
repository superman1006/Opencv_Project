import cv2  # opencv的缩写为cv2
import numpy as np  # numpy数值计算工具包


# 整体的操作流程包括四步操作是：
#       先对图像Sobel算子处理，再将结果取绝对值,再进行归一化,最后转换成uint8类型,
#       其中cv2.convertScaleAbs()操作包括了 [取绝对值] 和 [归一化] 和 [转换类型为uint8]  的操作。


# ① Sobel算子函数：cv2.Sobel(src, ddepth, dx, dy, ksize)，返回值为Sobel算子处理后的图像。
#       - ddepth：图像的深度,通常为 -1
#       - dx 和 dy 分别表示水平和竖直方向
#       - ksize 是 Sobel算子的大小,通常为 1,3,5,7
# ② 靠近最近点的左右和上下的权重最高，所以在Sobel.png中的矩阵的上下和左右比例为 ±2。
# 这些算子是用来计算图像中每个像素点的 梯度（灰度变化）近似值的，具体来说，就是用来求导数或者求图像的梯度。


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# -----------------------------------------------1. Sobel算子 --------------------------------------------------
Sobel = cv2.imread('references/Sobel.png')
cv_show(Sobel, 'Sobel')

# -----------------------------------------------1.1 圆形处理(例) -----------------------------------------------
pie = cv2.imread('references/pie.png')  # 读取图像
cv_show(pie, 'pie')

# (Sobel.png矩阵中数值代表是右边的RGB减左边别的RGB)，白为 255，黑为 0
# 白到黑是整数，黑到白是负数了，所有的负数会被截断成 0，所以要取绝对值
sobelx = cv2.Sobel(pie, cv2.CV_64F, 1, 0, ksize=3)

# 1,0 表示只算 x 水平方向梯度，不算 y 竖直方向梯度
cv_show(sobelx, 'sobelx1')

sobelx = cv2.convertScaleAbs(sobelx)  # 取负数时，取绝对值
cv_show(sobelx, 'sobelx2')

sobely = cv2.Sobel(pie, cv2.CV_64F, 0, 1, ksize=3)
# 0, 1只算 y 竖直方向梯度
sobely = cv2.convertScaleAbs(sobely)  # 取负数时，取绝对值
cv_show(sobely, 'sobely')

# 计算 x 和 y 后，再通过addWeighted求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# 0是偏置项
cv_show(sobelxy, 'sobelxy')

# 不建议直接计算,还有重影
sobelxy = cv2.Sobel(pie, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show(sobelxy, 'sobelxy')

# -----------------------------------------------1.2 人照处理(例) -----------------------------------------------

# 1.计算 x 和 y 后，再通过addWeighted求和(推荐用这种方法)
people = cv2.imread('references/people.png', cv2.IMREAD_GRAYSCALE)
cv_show(people, 'img')
sobelx = cv2.Sobel(people, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
# 这里不需要归一化，因为已经归一化了，归一化的操作在convertScaleAbs()中已经实现了
sobely = cv2.Sobel(people, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show(sobelxy, 'sobelxy')

# 2.整体计算有重影和模糊，不建议整体计算 XXX
sobelxy = cv2.Sobel(people, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show(sobelxy, 'sobelxy')

# -----------------------------------------------2. Scharr算子 --------------------------------------------------
Scharr = cv2.imread('references/Scharr.png')
cv_show(Scharr, 'Scharr')

# -----------------------------------------------3. Laplacian算子 -----------------------------------------------
Laplacian = cv2.imread('references/Laplacian.png')
cv_show(Laplacian, 'Laplacian')

# 三者的差别
people = cv2.imread('references/people.png', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(people, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(people, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(people, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(people, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(people, cv2.CV_64F)  # 没有 x、y，因为是求周围点的比较
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
# np.hstack()是将图像水平拼接，vstack()是将图像竖直拼接
cv_show(res, 'res')
