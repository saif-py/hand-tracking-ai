import cv2
import mediapipe as mp
import numpy as np


# brush thickness
# colour palette
# shapes
# letters


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=0, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        # if self.results.multi_hand_landmarks:
        #     # for handLms in self.results.multi_hand_landmarks:
        #     #
        #     #     if draw:
        #     #         self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            ms = lmlist[12]

            cs = lmlist[8]
            if draw:
                cv2.circle(image, (cs[1], cs[2]), 10, (255, 0, 25), cv2.FILLED)

        return lmlist


def main():
    xp, yp = 0, 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    fingers_up = []
    tipIds = [4, 8, 12, 16, 20]
    number_fingers_up = 0

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            cs = lmList[8]
            ms = lmList[12]
            # print(cs[1],cs[2],ms[1],ms[2])
            # print(lmList[8])-> coordinates of tip of index finger lmlist[8]
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers_up.append(tipIds[id])
                    # print(fingers_up)
                else:
                    pass
            # number of fungers up is len(fingers_up)
            # if number_fingers_up != fingers_up:  # to print the number of fingers up everytime it changes

            if len(fingers_up) != 5:

                pass
            # print(f'number of fingers_up is {len(fingers_up)}')
            else:
                if len(fingers_up) == 0:
                    if len(fingers_up) == 5:
                        print("eraser")
                    else:
                        pass
                else:
                    pass
            if len(fingers_up) == 2:
                # x2 , y2 = lmList[12][2:]
                cv2.circle(image, (ms[1], ms[2]), 10, (255, 0, 25), cv2.FILLED)
                # print(cs[1], cs[2], ms[1], ms[2])
                cv2.rectangle(image, (cs[1], cs[2] - 25), (ms[1], ms[2] + 25), (255, 0, 255), cv2.FILLED)
                # print('selection mode')
                drawing_mode = False
            if len(fingers_up) == 1:

                # if drawcolour ==(0,0,0):                                        #for eraser
                # cv2.line(imgCanvas,(xp,yp),(cs[1],cs[2]) , (0,0,0), 15)
                # cv2.line(image,(xp,yp),(cs[1],cs[2]) , (25,0,255), 15)     #(image, initial , final , color , brush size)

                # print('drawing mode')
                drawing_mode = True
                if xp == 0 and yp == 0:  # otherwise will start the line from (0,0)
                    xp, yp = cs[1], cs[2]
                cv2.line(image, (xp, yp), (cs[1], cs[2]), (25, 0, 255),
                         15)  # (image , initial pt , final pt , color , brush thickness)
                cv2.line(imgCanvas, (xp, yp), (cs[1], cs[2]), (255, 0, 255), 10)
                xp, yp = cs[1], cs[2]
                number_fingers_up = fingers_up
            if len(fingers_up) == 4:  # eraser
                # print("eraser mode")
                drawing_mode = False
                cv2.line(image, (xp, yp), (cs[1], cs[2]), (25, 255, 0), 15)
                cv2.line(imgCanvas, (xp, yp), (cs[1], cs[2]), (0, 0, 0), 100)


            xp, yp = cs[1], cs[2]
            # cv2.rectangle(image, (cs[1],cs[2]-25), (ms[1],ms[2]+25) , (25,0,255), cv2.FILLED )

            fingers_up = []

        # image = cv2.addWeighted(image,0.5,imgCanvas,0.5,0)  #transperancy
        # imgGRAY = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
        # _, imginv = cv2.threshold(imgGRAY,50,255,cv2.THRESH_BINARY_INV)
        # imginv = cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
        # image = cv2.bitwise_and(image,imginv)
        # image = cv2.bitwise_or(image,imgCanvas)
        # image = cv2.resize(image, (1280, 720))
        imgCanvas = cv2.resize(imgCanvas, (image.shape[1], image.shape[0]))
        ret, canvasM = cv2.threshold(imgCanvas[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        image2 = image.copy()
        image2[np.where(canvasM == 255)] = imgCanvas[np.where(canvasM == 255)]

        cv2.imshow('canvasM', canvasM)
        # cv2.imshow("Video", image)
        cv2.imshow("Video2", image2)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Canvas", imgCanvasr)

        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    image.relase()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
main()
