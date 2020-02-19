import numpy as np
import cv2
import math
import threading
from networktables import NetworkTables

'''
Note for other coders
The calibration need to be tuned for other backgrounds
The yellow boxes are the calibration objects (the one with the largest area
Prints the statements of the calibration object
Blue boxes are the tracked object
'''

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://10.36.36.216:4747/mjpegfeed?640x480')

# Rescaling function
def rescale_frame(res, percent=75):
    width = int(res.shape[1] * percent/ 100)
    height = int(res.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(res, dim, interpolation =cv2.INTER_AREA)

# Lemon finder function
def lemonFinder(frame):

    # to avoid error
    crop = frame
    
    # Converting image to hs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Blur hsv
    hsv = cv2.GaussianBlur(hsv, (41, 41), 0) 
    
    
    # Define range of yellow  color in HSV
    lower_yellow = np.array([21, 90, 15])  # old: [21,60,15]
    upper_yellow = np.array([35, 255, 255])  # old: [32,255,255]

    # Find yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    
    # Bounding boxes
    # For drawing img
    # Find contours
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    
    # Create variable for number of balls
    balls = 0
        
    # Variables for picking closest ball
    trackX = 0
    trackY = 0
    trackW = 2
    trackH = 2
    trackA = 1
    
    # Run for every contour
    for cont in contours:
        # Find boxes
        x, y, w, h = cv2.boundingRect(cont)

        # centers
        centerX = x + (w / 2)
        centerY = y + (h / 2)

        # Finding closest ball
        if (w * h) > trackA:
            trackX = x
            trackY = y
            trackA = (w * h)
            trackW = w
            trackH = h

        # for calibrating the tracker
        # crop image
        xpos = trackX + trackW
        ypos = trackY + trackH
        crop = frame[trackY:ypos, trackX:xpos]

        # average the crop color
        average1 = np.mean(crop, axis=0)
        average2 = np.mean(average1, axis=0)

        # Draw rectangle
        cv2.rectangle(frame, (trackX, trackY), (trackX + trackW, trackY + trackH), (10, 255, 255), 2)

        print(average2[0] > 30 and average2[0] < 80, average2[1] > 70 and average2[1] < 115, average2[2] > 65 and average2[2] < 160)

        print(average2[2])

        # actual tracking
        # crop image
        xpos = x + w
        ypos = y + h
        crop = frame[y:ypos, x:xpos]

        # average the crop color
        average1 = np.mean(crop, axis=0)
        average2 = np.mean(average1, axis=0)

        # use average to confirm ball
        if average2[0] > 30 and average2[0] < 80 and average2[1] > 70 and average2[1] < 115 and average2[2] > 65 and average2[2] < 160:

            # ball count
            balls += 1

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 100, 100), 2)

            # Add lemon position text
            cv2.putText(frame, 'Ball' + str(balls), (x + 5, int(y + (h/4))), 1, 1, 255, 2)
            cv2.putText(frame, 'Pos:', (x + 5, int(y + 2 * (h/4))), 1, 1, 255, 2)
            cv2.putText(frame, '(' + str(x) + ', ' + str(y) + ')', (x + 5, int(y + 3 * (h/4))), 1, 1, 255, 2)
            
            
    # resize the image
    # frame = rescale_frame(frame, 300)
    
    # Return frame and mask
    return(frame)



while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Run the lemon finder function
    frame = lemonFinder(frame)
    
    # Show the frames
    cv2.imshow('coloredCircles', frame)
    
    # If escape key is pressed exit while loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    

# Close the windows
cap.release()
cv2.destroyAllWindows()
