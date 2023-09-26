import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

#Calculating fps
previoustime=0
currenttime=0


while True:
    ret,img=cap.read()
    imgBGR=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgBGR)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handslandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handslandmark.landmark):
                print(id,lm)
                
            mpDraw.draw_landmarks(img,handslandmark,mpHands.HAND_CONNECTIONS)

    currenttime = time.time() 
    fps = 1 / (currenttime - previoustime)
    previoustime = currenttime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 139), 2)
 

    cv2.imshow('image',img)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()    