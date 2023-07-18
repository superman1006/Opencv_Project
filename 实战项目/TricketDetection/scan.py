# 导入工具包
import argparse
import cv2
import myutils


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# 读取输入
img_origin = cv2.imread(args["image"])
cv_show('img_origin', img_origin)

# 坐标也会相同变化
print('图片的高,宽和通道数为:', img_origin.shape)
img_copy = img_origin.copy()
img = myutils.resize(img_copy, height=500)
cv_show('img_resized', img)


# 预处理
print("STEP 1: 边缘检测")

#   1.转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

#   2.高斯模糊
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blur', blur)

#   3.Canny边缘检测
edged = cv2.Canny(blur, 75, 200)
cv_show('Edged', edged)


# 轮廓检测
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# sorted(),按照每一个轮廓的面积进行排序,再进行反转,所以变为从大到小排序,最后取前五个面积最大的轮廓,因为不一定只有一个小票,虽然这里只有一个
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
screenCnt = None
for c in contours:
    # 计算轮廓近似
    # cv2.arcLength()计算轮廓周长
    peri = cv2.arcLength(c, True)
    # C表示轮廓,第二个参数epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数,True表示封闭的
    # 第二个参数 越小,表示越精确,越大越规矩，方正
    # 对当前轮廓进行cv2.approxPolyDP()来计算近似的多边形曲线，并且返回每一个轮廓点
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print('approx:', approx)
    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        # 由于我们只需要检测到一个小票,所以找到一个就可以退出循环了
        break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv_show('Outline', img)

# 计算高度的比例，待会需要等比例缩放
ratio = img_origin.shape[0] / 500.0

# 透视变换
cv_show('img_copy', img_copy)
warped = myutils.four_point_transform(img_copy, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('result/scanned.jpg', ref)

# 展示结果
print("STEP 3: 变换")
cv_show('Original', myutils.resize(img_copy, height=650))
cv_show('Scanned', myutils.resize(ref, height=650))



















