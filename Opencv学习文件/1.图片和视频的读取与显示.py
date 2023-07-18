import cv2
import numpy as np
import matplotlib.pyplot as plt


#  图像的读取与显示
def cv_show(name, img):
    print(img)  # 用opencv读取图片是BGR格式
    print(type(img))  # 图片的类型
    print(img.shape)  # 图片的长宽以及BGR通道数
    print(img.size)  # 图片的像素点的个数
    cv2.imshow(name, img)  # 显示图片
    cv2.waitKey(0)  # 等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()  # 释放窗口
    print('-' * 50)


img1 = cv2.imread('references/Desktop1.png')
cv_show('image', img1)

img2 = cv2.imread('references/Desktop1.png', cv2.IMREAD_GRAYSCALE)  # 以灰度图的方式读取图片
cv_show('image', img2)

cv2.imwrite("results/test_gray_scale.png", img2)  # 保存图片到本地


#  视频的读取与显示

vc = cv2.VideoCapture('references/test.mp4')  # 读取视频
# VideoCapture()中参数是0，表示打开笔记本的内置摄像头

# 检查视频是否打开
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False

while open:
    ret, frame = vc.read()
    # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        cv2.imshow('result', gray)
        if cv2.waitKey(10) & 0xFF == 27:
            # waitKey()方法本身表示等待键盘输入，
            #   参数是1，表示延时1ms切换到下一帧图像
            #   参数为0，如cv2.waitKey(0)
            #   只显示当前帧图像，相当于视频暂停,
            #   参数过大如cv2.waitKey(1000)，会因为延时过久而卡顿感觉到卡顿。
            #   0xFF是键盘输入的ASCII码，esc键对应的ASCII码是27
            break










