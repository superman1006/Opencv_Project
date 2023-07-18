import cv2
import mediapipe as mp
import time
#  这里将刚刚写好的HandTrackingModule.py文件导入进来
import HandTrackingModule as htm


cv2.namedWindow("Camera", 0)
cv2.resizeWindow("Camera", 1200, 900)

# 相当于在这个文件中执行 HandTrackingModule 文件中的 main()函数
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Camera", img)
    c = cv2.waitKey(1)
    if c == 27:  # EXC键ASCII码：27
        break
cap.release()
cv2.destroyAllWindows()
