import numpy as np
import cv2
import math

cap = cv2.VideoCapture(1)

# Rescaling function
def rescale_frame(res, percent=75):
            width = int(res.shape[1] * percent/ 100)
            height = int(res.shape[0] * percent/ 100)
            dim = (width, height)
            return cv2.resize(res, dim, interpolation =cv2.INTER_AREA)

# Lemon finder function
def lemonFinder(frame):
    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mirroring the frame and hsv
    flippedhsv = cv2.flip( hsv, 1 )
    flippedframe = cv2.flip( frame, 1 )
    
    # Define range of yellow  color in HSV
    lower_yellow = np.array([21,60,15])
    upper_yellow = np.array([32,255,255])

    # Find yellow
    mask = cv2.inRange(flippedhsv, lower_yellow, upper_yellow)
    
    # Cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Find circles on binary image
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,1,20,param1=50,param2=15,minRadius=0,maxRadius=0)
    
    # Run circle code if error returned run except
    try:
        circles = np.uint16(np.around(circles))
        
        #Display the resulting frame with cicles
        for i in circles[0,:]:
            
            # Display circles
            cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),2)
            
            
    # When error returned 
    except:
        
        # Display text No Circles on screen
        cv2.putText(mask, "No Circles", (10, 70), 1, 2, 255, 2)
        
    # Return frame and mask
    return(flippedframe, mask)



while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Run the lemon finder function
    flippedframe, mask = lemonFinder(frame)    
    
    # Show the frames    
    cv2.imshow('coloredCircles', flippedframe)
    cv2.imshow('binaryImg', mask)
    
    # If escape key is pressed exit while loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    

# Close the windows
cap.release()
cv2.destroyAllWindows()
