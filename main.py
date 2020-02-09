import numpy as np
import cv2
import math
import threading
from networktables import NetworkTables

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('http://10.36.36.216:4747/mjpegfeed?640x480')   
         
def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

cond = threading.Condition()
notified = [False]
NetworkTables.initialize(server='10.36.36.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

# Insert your processing code here
print("Connected!")

# Rescaling function
def rescale_frame(res, percent=75):
            width = int(res.shape[1] * percent/ 100)
            height = int(res.shape[0] * percent/ 100)
            dim = (width, height)
            return cv2.resize(res, dim, interpolation =cv2.INTER_AREA)


def uploadPosition(x, y):
    sd = NetworkTables.getTable('SmartDashboard')
    sd.putNumber('X', x);
    sd.putNumber('Y', y)

# Lemon finder function
def lemonFinder(frame):
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
    #mask = cv2.dilate(mask, kernel, iterations=1)
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
            
            # Finding closest ball
            if (w * h) > trackA:
                trackX = x + (w / 2)
                trackY = y + (h / 2)
                trackA = (w * h)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w,y + h), (200,100,100), 2)
            
            # Add lemon position text
            cv2.putText(frame, 'Ball' + str(balls), (x + 5, y + (h/4)), 1, 1, 255, 2)
            cv2.putText(frame, 'Center:', (x + 5, y + 2 * (h/4)), 1, 1, 255, 2)
            cv2.putText(frame, '(' + str(x + (w/2)) + ', ' + str(y + (h/2)) + ')', (x + 5, y + 3 * (h/4)), 1, 1, 255, 2)
            
    # resize the image
    #frame = rescale_frame(frame, 300)
    
    
    # Old lemon finding code
    '''
    # Find circles on binary image
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,1,20,param1=50,param2=15,minRadius=0,maxRadius=0)
    
    # Run circle code if error returned run except
    try:
        circles = np.uint16(np.around(circles))
        
        #Display the resulting frame with cicles
        for i in circles[0,:]:
            
            # Display circles
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            
    # When error returned 
    except:
        
        # Display text No Circles on screen
        cv2.putText(frame, "No Circles", (10, 70), 1, 2, 255, 2)
    
    # Set circleThing to amount of circles plus 15
#    circleThing = circleAmount/2+15
    '''
    
    # Uploading x & y of closest ball to roborio
    print('(' + str(trackX) + ', ' + str(trackY) + ')')
    uploadPosition(trackX, trackY)
    
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
