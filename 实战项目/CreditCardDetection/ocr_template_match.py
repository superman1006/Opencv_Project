# 导入工具包
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()  # 创建 ArgumentParser 实例对象
# 下面代码的-i参数是输入图像的路径，-t参数是模板图像的路径
# ap.add_argument()函数用来指定输入的参数:
#       其中第一个参数是参数的名字
#       第二个参数是参数的缩写
#       第三个参数是required=True表示该参数是必须的
#       第四个参数是help参数是该参数的提示信息
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A card")
args = vars(ap.parse_args())

# credit_card的第一个数字 指定了 信用卡的类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ------------------------------------------------------------处理模板图像------------------------------------------------------------
# 流程:
#   -->  cv2.imread()读取一个模板图像
#   -->     cv2.cvtColor(src)转为灰度图
#   -->        cv2.threshold(灰度图src, 阈值, maxVal, cv2.COLOR_BGR2GRAY)[1]转为二值图像
#   -->           cv2.findContours(二值图.copy(), 轮廓检索的mode, 轮廓逼近method)获得轮廓
#   -->              cv2.drawContours(原图的src, 上一步获得的轮廓, -1, 颜色, 线条宽度)在原图画出轮廓
#   -->                 myutils.sort_contours(轮廓, 排序的method) 将已有的轮廓进行排序
#   -->                    创建一个字典{}用于储存最终结果
#   -->                       for遍历排序后的轮廓
#   -->                       { 对每个轮廓使用cv2.boundingRect(轮廓)获得轮廓的外接矩形的(x,y,w,h)
#   -->                             对于每一个外接矩形都在二值图中裁剪下来得到每个数字的 ROI区域
#   -->                                将每个ROI区域resize成合适大小
#   -->                                   将每一个ROI区域(数字的最终的模板)和其对应的所以存入刚刚的字典中 }

# 读取一个模板图像,由于已经在上方已经设置了参数，所以这里直接读取
template_origin = cv2.imread(args["template"])
cv_show('template_origin', template_origin)

# 转为 灰度图
template_gray = cv2.cvtColor(template_origin, cv2.COLOR_BGR2GRAY)
cv_show('template_gray', template_gray)

# 再转为 二值图像
# cv2.THRESH_BINARY : 超过阈值部分取maxval ( 最大值 )，否则取 0 ,INV表示反转
template_thresh = cv2.threshold(template_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
# [1]表示只取返回的第二个参数，也就是二值图像，[0]表示只取第一个参数，也就是阈值ret
cv_show('template_thresh', template_thresh)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图, cv2.RETR_EXTERNAL只检测 外轮廓 , cv2.CHAIN_APPROX_SIMPLE只保留 终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
template_contours, hierarchy = cv2.findContours(template_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template_origin, template_contours, -1, (0, 0, 255), 3)
cv_show('template_draw_contours', template_origin)
print('template_contours的shape值为:', np.array(template_contours, dtype=object).shape)
template_contours, boundingBoxes = myutils.sort_contours(template_contours, method="left-to-right")  # 排序，从左到右，从上到下
# 使用myutils.sort_contours()将template_contours按照从左到右，从上到下的顺序进行排序,并且返回boundingBoxes(包含每一个轮廓的外接矩形的(x,y,h,w))
print('template中0~9对应的(x,y,h,w)为:\n', boundingBoxes)
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(template_contours):
    # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出 数据下标 和 该数据
    # 这里的i就是contour对应的index索引,c就是contour本身
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = template_thresh[y:y + h, x:x + w]
    # ROI区域是感兴趣区域，注意要先写y区域，再写x区域，因为数组的行对应y，列对应x
    roi = cv2.resize(roi, (57, 88))
    # 把模板调成 (57, 88) 大小,后面的模板匹配也是按照这个大小进行的

    # 存入每一个数字对应每一个模板
    digits[i] = roi

# ------------------------------------------------------------处理实例图像------------------------------------------------------------

# 流程:
#   -->  cv2.imread()读取一个实例图像
#   -->    myutils.resize(src, width, height)调整图像大小
#   -->      cv2.cvtColor(src)转为灰度图
#   -->        cv2.getStructuringElement(形状, 大小)创建两个合适大小的卷积核
#   -->          cv2.morphologyEx(灰度图src, cv2.MORPH_TOPHAT, kernel)进行 tophat 形态学操作
#   -->            sobel操作1: cv2.Sobel(tophat图, cv2.CV_8U, dx, dy, ksize)用来检测图像的水平梯度
#   -->            sobel操作2: np.absolute(gradX)取绝对值
#   -->            sobel操作3: ((梯度 - min梯度) / (max梯度 - min梯度) * 255)将梯度图像归一化
#   -->            sobel操作4: 梯度.astype("uint8")转为uint8类型
#   -->              cv2.morphologyEx(上一步得到的梯度图, cv2.MORPH_CLOSE, kernel)进行闭操作让数字更加连续在一起
#   -->                cv2.threshold(上一步得到的梯度图, 阈值, maxVal, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]转为二值图像
#   -->                  (如果上一步的闭操作已经很好了，这一步可以省略)cv2.morphologyEx(上一步得到的二值图, cv2.MORPH_CLOSE, 另一个kernel)再进行一次闭操作
#   -->                    cv2.findContours(二值图.copy(), 轮廓检索的mode, 轮廓逼近method)获得轮廓
#   -->                      (画的操作可以省略,因为只是方便展示流程而已)cv2.drawContours(原图的copy, 上一步获得的轮廓, -1, 颜色, 线条宽度)在原图的copy画出轮廓
#   -->                        创建一个列表[]用于储存示例图片中的每个数字的坐标
#   -->                          for遍历排序后的轮廓{
#   -->                                 对每个轮廓使用cv2.boundingRect(轮廓)获得轮廓的外接矩形的(x,y,w,h)
#   -->                                 计算每一个轮廓的宽和高的比例，通过宽,高和宽高比，筛选出目标轮廓(也就是四个数字为一组的轮廓)
#   -->                                 将筛选后得到的的目标轮廓的(x,y,w,h)存入列表中
#   -->                          }
#   -->                          通过sorted()将刚刚的loc列表按照x坐标从小到大排序
#   -->                          创建一个output列表用来存储识别后输出的每个数字
#   -->                          for遍历排序后的loc列表{
#   -->                                  创建一个临时列表用于存储每一个loc检测到的数字和模板匹配后 score最高的那个数字(每一个 临时列表 将会储存 4 个数字)
#   -->                                  根据每一个loc中的(x,y,w,h)在原图的灰度图上截取出适当的ROI区域
#   -->                                  cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]转成适当的二值图像
#   -->                                  cv2.findContours()获得一组数字的轮廓
#   -->                                  myutils.sort_contours()对这组数字的轮廓从左到右进行排序
#   -->                                  for遍历 刚刚整理好的那一组轮廓，取出每一个轮廓{
#   -->                                          通过cv2.boundingRect()获得每一个数字的外接矩形的(x,y,w,h)
#   -->                                          裁剪出每一个数字的ROI区域
#   -->                                          cv2.resize()将ROI调整成和上方模板resize()后一样的大小
#   -->                                          创建一个scores空列表用于存储当前组中每一个数字和模板匹配后的score
#   -->                                          for遍历 模板digits列表中的每一个模板和其对应的数字{
#   -->                                                  cv2.matchTemplate()进行模板匹配
#   -->                                                  cv2.minMaxLoc(匹配后的结果) 获得最大值并且存入scores列表中
#   -->                                          }
#   -->                                          通过np.argmax()获得scores列表中最大值的索引, 从而得到识别后认为最匹配的数字
#   -->                                   }
#   -->                                   cv2.rectangle(img, 左上角坐标、右下角坐标、线框颜色、线框粗细) 在原图上画出一个 长方形
#   -->                                   cv2.putText(原图img, 文字内容:string, 文字左下角坐标, 字体, 字体大小, 字体颜色, 字体粗细)在原图上写当前组识别后的数字结果
#   -->                          }


# 读取输入图像，预处理
card = cv2.imread(args["image"])
cv_show('card', card)
card = myutils.resize(card, width=300)
card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
cv_show('card_gray', card_gray)

# 初始化卷积核
# cv2.getStructuringElement()用于构造一个特定形状的结构元素，第一个参数是形状，第二个参数是尺寸
# 创建两个不同大小的卷积核 (9X3更好用来处理单个数字因为每个数字的 h 和 w 的比例大约为9:3)
# rectKernel中的rect表示rectangle,也就是矩形
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# sqKernel中的sq表示square,也就是正方形
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 礼帽 形态学操作，突出更明亮的区域，tophat操作作用是将图片中的 亮度 高于周围的区域都突出出来
tophat = cv2.morphologyEx(card_gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 通过Sobel算子计算 X 方向上的梯度, grad是 梯度 的意思
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1 相当于用 3*3 的 Sobel算子
gradX = np.absolute(gradX)  # 取绝对值
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 计算最小值和最大值

# 进行归一化，因为后面要用这个梯度值进行阈值分割
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化,将梯度图像映射到[0,255]区间
gradX = gradX.astype("uint8")  # 转换为uint8类型，方便后面进行阈值操作

print('gradX的shape:', np.array(gradX).shape)
cv_show('gradX', gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起,这样四个数字就会被连在一起！！！！！
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)

# THRESH_OTSU 会自动寻找合适的 阈值，适合双峰，需把阈值参数设置为0
# 这里使用0和255并不是把阈值设置为0，而是让阈值自动寻找合适的阈值,生成合适的二值图像
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)

# 再来一个闭操作
# 这里使用sqKernel，因为数字之间的间隙比较小
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh_twice', thresh)

# 计算轮廓
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
copy_img = card.copy()
cv2.drawContours(copy_img, cnts, -1, (0, 0, 255), 2)
cv_show('draw_img', copy_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 筛选合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if 2.5 < ar < 4.0:

        if (40 < w < 55) and (10 < h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))
print('符合的轮廓locs:', locs)
# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []  # 用于存储 card图片中检测到的数字和模板匹配后 score最高的那个数字

    # 根据坐标提取每一个组,往上多取5个像素，往下多取5个像素，这样就包含了数字上下的一点空隙
    group = card_gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]  # 因为图片是矩阵，所以这里的坐标是反的！！！！！
    cv_show(f'第{i}组', group)
    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 计算每一组的轮廓
    card_group_digit_cnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    card_group_digit_cnts = myutils.sort_contours(card_group_digit_cnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in card_group_digit_cnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit_value, template_digit) in digits.items():  # digits.items()返回的是一个元组，元组中的元素是键值对
            # (digit_value, template_digit)分别指模板的 数字 和 模板图片
            # 模板匹配
            result = cv2.matchTemplate(roi, template_digit, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, _) = cv2.minMaxLoc(result)
            # 不关心其他的数值，只关心最大值，所以只返回最大值也就是 我们所需要的 score
            scores.append(maxVal)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    # cv2.rectangle()用于在图片上绘制矩形
    # 参数分别是：原图、矩形的左上角坐标、矩形的右下角坐标、线框颜色、线框粗细
    cv2.rectangle(card, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)

    # cv2.putText()用于在图片上绘制文字，参数分别是：原图、文字内容、文字左下角坐标、字体、字体大小、字体颜色、字体粗细
    # "".join(groupOutput)将groupOutput中的元素拼接成字符串，并且每个元素中间用""隔开
    cv2.putText(card, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv_show('card_matched', card)
