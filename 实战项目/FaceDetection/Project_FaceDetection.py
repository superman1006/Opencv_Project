import cv2
import mediapipe as mp
import time
#  这里将刚刚写好的FaceDetectionModule.py文件导入进来
import FaceDetectionModule as fdm


cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", 0)
cv2.resizeWindow("Camera", 1200, 900)
cTime = 0
pTime = 0
detector = fdm.FaceDetector()
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    print(bboxs)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText()的参数：图像，文本，文本位置，字体，字体大小，颜色，字体粗细
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
