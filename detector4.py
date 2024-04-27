import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
def click_event(event, x, y, flags, param):
    global draw, a, b
    if event == cv2.EVENT_LBUTTONDOWN:
        a, b = x, y
        draw = 1
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == 1:
            cv2.rectangle(frame1, (a, b), (x, y), (0, 0, 0), 1)
            cv2.imshow("frame", frame1)
            #cv2.waitKey(1)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(frame1, (a, b), (x, y), (0, 0, 255), 1)
        global rect
        rect = a, b, x, y
        draw = 0
        cv2.putText(frame1, 'Press any key to continue', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame1)
        #cv2.waitKey(1)

global draw, frame1, rect
draw = 0
vcap = cv2.VideoCapture("video7.mp4")
ret, frame1 = vcap.read()
rect = 0, 0, frame1.shape[1], frame1.shape[0]
cv2.imshow("frame", frame1)
cv2.setMouseCallback('frame', click_event)
cv2.waitKey(0)
cv2.destroyWindow('frame')

# you can set custom kernel size if you want
kernel = np.ones((10,10),np.uint8)

# initilize background subtractor object (1)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50, history=2800)

# Noise filter threshold
thresh = 1500

while (1):
    ret, frame = vcap.read()
    if not ret:
        break
    forbid = frame[rect[1]:rect[3], rect[0]:rect[2]]
    # Apply background subtraction, isolating moving objects (displayed as white) (2)
    fgmask = backSub.apply(forbid)

    # Get rid of the shadows (3)
    ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Apply some morphological operations to make sure you have a good mask (4)
    fgmask = cv2.erode(fgmask, kernel, 200)
    #fgmask = cv2.dilate(fgmask, kernel, iterations=4)

    # Detect contours in the frame (5)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        # Get the maximum contour
        cnt = max(contours, key=cv2.contourArea)

        # make sure the contour area is somewhat hihger than some threshold to make sure its a person and not some noise.
        if cv2.contourArea(cnt) > thresh:

            # Draw a bounding box around the person and label it as person detected (6)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(forbid, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(forbid, 'Person Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

    # Stack both frames and show the image
    fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Res', frame)
    #cv2.imshow('Res', fgmask_3)
    #stacked = np.hstack((fgmask_3, frame))
    #cv2.imshow('Combined', cv2.resize(stacked, None, fx=0.65, fy=0.65))

    if cv2.waitKey(30) == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()