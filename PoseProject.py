import cv2
import mediapipe as mp
import time
import PoseModule as pm

cap = cv2.VideoCapture(r'C:\Users\Yachna Hasija\PycharmProjects\PoseEstimation\Videos\4.mp4')
pTime = 0
detector = pm.posedetector()
while (True):
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findposition(img, draw = False)
    if (len(lmlist)!=0):
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 30, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (100, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 10)
    cv2.imshow('image', cv2.resize(img, (800, 600)))
    key = cv2.waitKey(5)
    if key == ord('q'):
        cv2.destroyAllWindows()