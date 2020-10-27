# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:30:17 2020

@author: x
"""

import sys
import cv2 as cv
import numpy as np

def nothing(x):
    pass

capturing = True

cap = cv.VideoCapture("szukaj_zielonego.webm")

if cap.isOpened() == False:
    print("error opening frames source")
    exit()

if len(sys.argv) == 3:
    dim = (int(sys.argv[1]),int(sys.argv[2]))
else:
    dim = (320,200)

print("Video size: {} x {}".format(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

sliders_window = "sliders"
cv.namedWindow(sliders_window)
cv.createTrackbar("H upper", sliders_window, 255, 255, nothing)
cv.createTrackbar("H lower", sliders_window, 0, 255, nothing)
cv.createTrackbar("S upper", sliders_window, 255, 255, nothing)
cv.createTrackbar("S lower", sliders_window, 0, 255, nothing)
cv.createTrackbar("V upper", sliders_window, 255, 255, nothing)
cv.createTrackbar("V lower", sliders_window, 0, 255, nothing)    

while True:
    
    success, frame = cap.read()
    key = cv.waitKey(33)
    
    if success:
        frame = cv.resize(frame, dim)
        frame = cv.flip(frame, 1)
        frame = cv.GaussianBlur(frame, (5,5), 0)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hu = cv.getTrackbarPos("H upper","sliders")
        hl = cv.getTrackbarPos("H lower","sliders")
        su = cv.getTrackbarPos("S upper","sliders")
        sl = cv.getTrackbarPos("S lower","sliders")
        vu = cv.getTrackbarPos("V upper","sliders")
        vl = cv.getTrackbarPos("V lower","sliders")
        upper = np.array([hu,su,vu])
        lower = np.array([hl,sl,vl])
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("Mask", mask)
        cv.imshow("Result", result)
        cv.imshow("Not-yet smart window", frame)
        hsv = cv.putText(hsv, "H: {} - {}".format(hl, hu), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        hsv = cv.putText(hsv, "S: {} - {}".format(sl, su), (0, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        hsv = cv.putText(hsv, "V: {} - {}".format(vl, vu), (0, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow("HSV", hsv)
        if key == 120:
            r = cv.selectROI("Result", result)
            resultCrop = result[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            cv.imwrite("image.png", resultCrop)
    else:
        # stream finished
        capturing = False
    if key == 27:
        capturing = False
    if capturing == False:
        break

cap.release()
cv.destroyAllWindows()



