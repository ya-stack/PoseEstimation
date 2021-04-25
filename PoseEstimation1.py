import cv2
import mediapipe as mp
import time

mPose = mp.solutions.pose
pose =  mPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap =  cv2.VideoCapture(r'C:\Users\Yachna Hasija\PycharmProjects\PoseEstimation\Videos\1.mp4')
pTime = 0

while (True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 10, (255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (100,150), cv2.FONT_HERSHEY_PLAIN,10 , (255,0,255), 10)
    cv2.imshow('image', cv2.resize(img, (800, 600)))
    key = cv2.waitKey(5)
    if key==ord('q'):
        cv2.destroyAllWindows()