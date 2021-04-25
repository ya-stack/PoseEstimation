import cv2
import mediapipe as mp
import time

class posedetector():

    def __init__(self, mode=False, upbody=False, smooth=True, detection_confidence=0.5,
                 tracconfidence=0.5):
        self.mode = mode
        self.upbody = upbody
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracconfidence = tracconfidence
        self.mPose = mp.solutions.pose
        self.pose = self.mPose.Pose(self.mode, self.upbody,self.smooth, self.detection_confidence,
                                    self.tracconfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mPose.POSE_CONNECTIONS)
        return img

    def findposition(self, img, draw = True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                if draw:
                    # if id == 4:
                    cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
        return lmlist

def main():
    cap = cv2.VideoCapture(r'C:\Users\Yachna Hasija\PycharmProjects\PoseEstimation\Videos\4.mp4')
    pTime = 0
    detector = posedetector()
    while (True):
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findposition(img, draw = False)
        if len(lmlist)!=0:
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

if __name__ == '__main__':
    main()