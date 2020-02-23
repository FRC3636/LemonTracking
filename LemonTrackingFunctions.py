import cv2
import numpy as np

# class for lemon tracking
class TrackingFunctions:

    def __init__(self):
        # variables
        self.frame = None
        self.hsv = None
        self.mask = None
        self.dist_img = None

    def IsLemon(average, w, h):
        # confirms lemon
        return average[0] > 20 and average[0] < 30 and average[1] > 80 and average[1] < 220 and average[2] > 90 and average[2] < 220 and (w * h) > 500

    def centerCrop(img, w, h):
        # crops around center
        maxY, maxX = img.shape[0], img.shape[1]
        x1, y1 = int(maxX / 2 - w / 2), int(maxY / 2 - h / 2)
        x2, y2 = int(maxX / 2 + w / 2), int(maxY / 2 + h / 2)
        crop = img[y1:y2, x1:x2]
        return crop

    def circleAverage(self, x, y, w, h):
        # get average of the crop in a given radius of w/3
        xpos = x + w
        ypos = y + h
        radius = w / 3
        crop = self.hsv[y:ypos, x:xpos]
        circleMask = self.centerCrop(self.dist_img, w, h) < radius ** 2
        circleColor = crop[circleMask]
        return np.mean(circleColor, axis=0)

    def FrameSetup(self):
        # Converting image to hsv and blur
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.hsv = cv2.GaussianBlur(self.hsv, (41, 41), 0)

    def Mask(self):
        # Define range of yellow  color in HSV and mask img
        lower_yellow = np.array([21, 50, 15])  # old: [21,60,15]
        upper_yellow = np.array([35, 255, 255])  # old: [32,255,255]
        self.mask = cv2.inRange(self.hsv, lower_yellow, upper_yellow)

        # Cleaning up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.mask = cv2.erode(self.mask, kernel, iterations=3)

    def FindAndProcessContours(self, showColorAverage=False):
        # Create variable to keep track of the number of lemons and their centers and area
        lemons = []

        # Find contours
        contours, hierarchy = cv2.findContours(self.mask, 1, 2)

        # Run for every contour
        for cont in contours:
            # Find boxes
            x, y, w, h = cv2.boundingRect(cont)

            average = self.circleAverage(self, x, y, w, h)

            # used for calibration
            if showColorAverage:
                print('Average: %s    Area: %s' % (average, (w * h)))

            # use average to confirm ball
            if self.IsLemon(average, w, h):

                # update lemon count
                lemons.append([x + (w / 2), y + (h / 2), w * h])

                # for scaling text
                scale = w / 80
                if w > 120:
                    scale = 1.5

                # Draw bounding box and text
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (200, 50, 50), 2)
                cv2.putText(self.frame, 'Lemon%s' % len(lemons), (x, int(y - h / 50)), 1, scale, (200, 50, 50), 2)

        return lemons