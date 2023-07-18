import cv2
import numpy as np


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # boundingRect()函数是用来计算轮廓的最小外接矩形的，接收单个contour，返回一个tuple(x,y,h,w)，包含矩形左上角的x,y坐标以及矩形的高和宽
    # boundingBoxes 为一个二维tuple,这个tuple包含10个小tuple
    # 把输入的每个 contour 用一个最小的矩形,包起来,(x,y,h,w),x,y是矩形左上角的坐标,h,w是矩形的高和宽

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # zip( [1,2,3], [4,5,6] ) 函数是将两个列表对应位置的元素打包为一个小tuple，然后存到一个大列表中,如[(1, 4), (2, 5), (3, 6)]
    # zip(cnts, boundingBoxes)之后的结果应该类似于这样：[(cnt1, boundingBox1), (cnt2, boundingBox2), ...]

    # sorted(iterable, cmp=None, key=None, reverse=False)  可以对所有可迭代的对象进行排序操作，而不仅仅是列表
    #   iterable -- 可迭代对象。
    #   cmp      -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回 - 1，等于则返回 0
    #   key      -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    #   reverse = True 降序 ， reverse = False 升序

    # ”lambada b“ 中的 b就是zip(cnts, boundingBoxes)中的每一个元素，也就是要按照 b[1][i]来sort排序
    # 举个例子:如果i=0,那么b[1][i]就是boundingBoxes中的每一个元素的第一个元素，也就是x坐标，按照x坐标来排序
    #         如果i=1,那么b[1][i]就是boundingBoxes中的每一个元素的第二个元素，也就是y坐标，按照y坐标来排序
    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
