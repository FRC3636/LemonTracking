import numpy as np
import cv2
import math

cap = cv2.VideoCapture(1)

while(True):
    def rescale_frame(res, percent=75):
            width = int(res.shape[1] * percent/ 100)
            height = int(res.shape[0] * percent/ 100)
            dim = (width, height)
            return cv2.resize(res, dim, interpolation =cv2.INTER_AREA)

    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of yellow  color in HSV
    lower_yellow = np.array([16,80,80])
    upper_yellow = np.array([30,255,255])

    #find yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #show the found yellow as yellow
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #flip because mirror is nicer
    res = cv2.flip(res,1)

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,200,param1=1000,param2=5000,minRadius=0,maxRadius=100000)
    try:
        circles = np.uint16(np.around(circles))
    #Display the resulting frame with cicles
        for i in circles[0,:]:
            cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),2)
    except AttributeError:
    #scale image
      res500 = rescale_frame(res, percent=500)
    cv2.imshow('res500',res500)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    continue

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
