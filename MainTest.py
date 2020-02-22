import numpy as np
import cv2
import time

'''
Note for other coders
The calibration need to be tuned for other backgrounds
The yellow boxes are the calibration objects (the one with the largest area
Prints the statements of the calibration object
Blue boxes are the tracked object
'''

cap = cv2.VideoCapture(0)

def createCircle(w, h):
    # creates distance img for creating circular crops
    X, Y = np.ogrid[:w, :h]
    CenterX, CenterY = w/2, h/2
    img = ((X - CenterX) ** 2 + (Y - CenterY) ** 2)

    return img

def centerCrop(img, w, h):
    # crops around center
    maxY, maxX =  img.shape[0], img.shape[1]
    x1, y1 = int(maxX/2 - w/2), int(maxY/2 - h/2)
    x2, y2 = int(maxX/2 + w/2), int(maxY/2 + h/2)
    crop = img[y1:y2, x1:x2]
    return crop


# Rescaling function
def rescale_frame(res, percent=75):
    width = int(res.shape[1] * percent/ 100)
    height = int(res.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(res, dim, interpolation =cv2.INTER_AREA)

# Lemon finder function
def lemonFinder(frame, dist_img):

    # to avoid error
    crop = frame
    # Converting image to hsv and blur
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (41, 41), 0) 

    # Define range of yellow  color in HSV and mask img
    lower_yellow = np.array([20, 55, 15])  # old: [21,60,15]
    upper_yellow = np.array([35, 255, 255])  # old: [32,255,255]
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find bounding Boxes
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    
    # Create variable for number of balls
    balls = 0
    
    # Run for every contour
    for cont in contours:
        # Find boxes
        x, y, w, h = cv2.boundingRect(cont)

        # actual tracking
        # crop image
        xpos = x + w
        ypos = y + h
        crop = hsv[y:ypos, x:xpos]
        circleMask = centerCrop(dist_img, w, h) < (w /2) ** 2

        circleColor = crop[circleMask]

        # average the crop color
        average = np.mean(circleColor, axis=0)

        # Draw rectangle
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 2)

        print(average, (w * h))

        # use average to confirm ball
        if average[0] > 22 and average[0] < 30 and average[1] > 65 and average[1] < 175 and average[2] > 100 and average[2] < 175:

            # ball count
            balls += 1

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 100, 100), 2)

            # Add lemon position text
            cv2.putText(frame, 'Lemon' + str(balls), (x + 5, int(y + (h/4))), 1, 1, 255, 2)
            cv2.putText(frame, 'Pos:', (x + 5, int(y + 2 * (h/4))), 1, 1, 255, 2)
            cv2.putText(frame, '(' + str(x) + ', ' + str(y) + ')', (x + 5, int(y + 3 * (h / 4))), 1, 1, 255, 2)


    # resize the image
    frame = rescale_frame(frame, 300)
    
    # Return frame and mask
    return(frame, crop)


dist_img = createCircle(480, 640)

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Run the lemon finder function
    frame, crop = lemonFinder(frame, dist_img)
    
    # Show the frames
    cv2.imshow('crop', crop)
    cv2.imshow('coloredCircles', frame)
    
    # If escape key is pressed exit while loop
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Close the windows
cap.release()
cv2.destroyAllWindows()
