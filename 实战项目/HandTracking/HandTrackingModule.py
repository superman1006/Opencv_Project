import cv2
# mediapipe是谷歌开发的一款机器学习框架，其中包含了很多机器学习模型，比如人脸检测，手势识别等等
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, complexity_con=1, detectionCon=0.5, trackCon=0.5):
        # mode是一个布尔值，用于设置是否检测手部的方向，如果设置为True，那么检测到的手部的方向会显示在控制台中
        self.mode = mode
        # maxHands是一个整数，用于设置检测到的最大手部的数量
        self.maxHands = maxHands
        # complexityCon是复杂度的意思，这里设置为1，可以提高检测的准确率，但是会降低检测的速度
        self.complexityCon = complexity_con
        # detectionCon是检测的置信度，这里设置为0.5，也就是说，如果检测到的手部的置信度小于0.5，那么就不会被检测到
        self.detectionCon = detectionCon
        # trackCon是跟踪的置信度，这里设置为0.5，也就是说，如果跟踪到的手部的置信度小于0.5，那么就不会被跟踪到
        self.trackCon = trackCon
        # mpHands是一个类,可以用来检测手部的关键点,需要传入上面的5个参数
        self.mpHands = mp.solutions.hands
        # hands是一个对象，用于检测手部的关键点
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexityCon, self.detectionCon, self.trackCon)
        # mpDraw用于在图像上绘制检测到的手部的关键点
        self.mpDraw = mp.solutions.drawing_utils
        # results用于存储检测到的手部的关键点的信息(存储self.hands.process(imgRGB)处理后的结果)
        self.results = None

    def findHands(self, img, draw=True):
        # imgRGB是一个图像，用于存储从BGR格式转换为RGB格式的图像
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        # 如果检测到的手部的关键点不为空，那么就绘制检测到的手部的关键点
        if self.results.multi_hand_landmarks:
            # 遍历检测到的手部的关键点,lms代表landmarks,也就是关键点的意思
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 如果draw为True，则使用mpDraw.draw_landmarks()方法在图像上绘制检测到的手部的关键点
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        # lmList是一个列表，用于存储检测到的手部的关键点的信息
        lmList = []
        if self.results.multi_hand_landmarks:
            # 依次遍历每一个手部的关键点，myHand是一个对象，用于存储 当前这只手 的关键点的信息
            for hand in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        # cv2.circle()的参数：图像，圆心坐标，半径，颜色，填充
                        # 用于在图像上绘制圆形
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

        return lmList


def main():
    cv2.namedWindow("Camera", 0)
    cv2.resizeWindow("Camera", 1200, 900)

    # pTime是previous time的缩写，用于存储上一次检测到手部的时间
    pTime = 0
    # cTime是current time的缩写，用于存储当前检测到手部的时间
    cTime = 0
    # 使用cv2.VideoCapture()方法读取摄像头，参数为0表示读取电脑自带的摄像头，参数为1表示读取外接的摄像头,并且返回一个对象cap代表摄像头
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        # 使用cap.read()方法读取摄像头的图像，返回两个值，success一个布尔值，表示是否读取成功，第二个值是一个图像
        success, img = cap.read()
        # 调用findHands()方法，传入img,进行手部检测，返回一个已经绘制了手部关键点的图像
        img = detector.findHands(img)
        # 调用findPosition()方法，传入img,进行手部关键点的检测，返回一个列表，列表中的每一个元素都是一个列表，包含了每个手部关键点的id和坐标
        lmList = detector.findPosition(img)
        # 长度为零表示没有检测到手部
        if len(lmList) != 0:
            print('大拇指的ID和位置:', lmList[4])
        # 使用time.time()方法获取当前时间，赋值给cTime
        cTime = time.time()
        # 用于计算帧率
        fps = 1 / (cTime - pTime)
        # 将cTime赋值给pTime，用于下一次计算
        pTime = cTime
        # cv2.putText()的参数：图像，文本，文本位置，字体，字体大小，颜色，字体粗细
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Camera", img)

        if cv2.waitKey(1) == 27:  # EXC键ASCII码：27
            break
    # 释放摄像头
    cap.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
