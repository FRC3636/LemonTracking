import numpy as np
import cv2
import math
import threading
from networktables import NetworkTables
import Network

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://10.234.1.144:4747/mjpegfeed?640x480')

# Created Network object
networkObj = Network.Network()
   

# Lemon finder function
def lemonFinder(frame):

    # Set null for trackX and Y
    trackX = None
    trackY = None    
    
    # Converting image to hs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Blur hsv
    hsv = cv2.GaussianBlur(hsv, (41, 41), 0) 
    
    
    # Define range of yellow  color in HSV
    lower_yellow = np.array([25,60,15]) # old: [21,60,15]
    upper_yellow = np.array([35,255,255]) # old: [32,255,255]

    # Find yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    #mask = cv2.dilate(mask, kernel, iterations=4)
    
    
    # Bounding boxes
    # For drawing img
    # Find contours
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    
    # Create variable for number of balls
    balls = 0
        
    # Variables for picking closest ball
    trackA = 0
    
    # Run for every contour
    for cont in contours:
        # Find boxes
        x, y, w, h = cv2.boundingRect(cont)
        
        # Draw boxes large objects by using area to remove unwanted objects (like hair)
        if (w * h) > 500:
            balls += 1
            
            # X and y of center of ball
            centerX = x + (w / 2)
            centerY = y + (h / 2)

            
            # Finding width and height of shape
            screenHeight, screenWidth = frame.shape[:2]
            
            
            # Finding percentage of ball on screen
            centerXPercent = int((centerX / float(screenWidth)) * 100)
            centerYPercent = int((centerY / float(screenHeight)) * 100)
            
            #print(str(centerX) + ", " + str(centerY))
            print(str(centerXPercent) + ", " + str(centerYPercent))
            #print(str(screenHeight) + ", " + str(screenWidth))
            
            # Finding closest ball
            if (w * h) > trackA:
                trackX = centerXPercent
                trackY = centerYPercent
                trackA = (w * h)
                
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w,y + h), (200,100,100), 2)
            
            # Add lemon position text
            cv2.putText(frame, 'Ball' + str(balls), (x + 5, y + int((h/4))), 1, 1, 255, 2)
            cv2.putText(frame, 'Center:', (x + 5, y + 2 * int((h/4))), 1, 1, 255, 2)
            cv2.putText(frame, '(' + str(centerX) + ', ' + str(centerY) + ')', (x + 5, y + 3 * int((h/4))), 1, 1, 255, 2)
            
            
            
    # resize the image
    #frame = rescale_frame(frame, 300)
    
    # Uploading x & y of closest ball to roborio
    #print('(' + str(trackX) + ', ' + str(trackY) + ')')
    if(trackX is not None and trackY is not None):
        NetworkObj.uploadPosition(trackX, trackY)
        
    
    # Return frame and mask
    return(frame, mask)



while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Run the lemon finder function
    frame, mask = lemonFinder(frame)    
    
    
    # Resize the frame
    # frame = rescale_frame(frame, 200)
    
    # Show the frames
    cv2.imshow('coloredCircles', frame)
    #cv2.imshow('binaryImg', mask)
    
    # If escape key is pressed exit while loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    

# Close the windows
cap.release()
cv2.destroyAllWindows()
