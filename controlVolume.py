import cv2
import time
import numpy as np 
import Hand as htm

# 
widht, panjang = 640, 480

cap = cv2.VideoCapture(0)


cap.set(2, widht)
cap.set(3, panjang)

pTime = 0

while (True):
    success, img = cap.read()
    cTime = time.time()
    fps = 1/ ( cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0 ), 3)
    
    cv2.imshow("Img", img)
    cv2.waitKey(2)
    