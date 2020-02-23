import cv2
import LemonTrackingFunctions
import ImageFunctions
import numpy as np

ImgFuncs = ImageFunctions.Functions
TrackingFuncs = LemonTrackingFunctions.TrackingFunctions

# set capture device
cap = cv2.VideoCapture(0)

# Capture frame for distance img
ret, frame = cap.read()

# create distance img for circle masking
TrackingFuncs.dist_img = ImgFuncs.createCircle(frame.shape[0], frame.shape[1])

while(True):

    # Capture frame
    ret, TrackingFuncs.frame = cap.read()

    # Find the Lemon
    TrackingFuncs.FrameSetup(TrackingFuncs)
    TrackingFuncs.Mask(TrackingFuncs)
    lemons = TrackingFuncs.FindAndProcessContours(TrackingFuncs)

    scaledFrame = ImgFuncs.rescale_frame(TrackingFuncs.frame, 400)

    # Show the frames
    cv2.imshow('YellowMask', TrackingFuncs.mask)
    cv2.imshow('MainWindow', scaledFrame)

    # If escape key is pressed exit loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Close the windows
cap.release()
cv2.destroyAllWindows()