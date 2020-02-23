import cv2
import numpy as np

# Class of functions used for simple image processing
class Functions:
    def createCircle(w, h):
        # creates distance img for creating circular crops
        X, Y = np.ogrid[:w, :h]
        CenterX, CenterY = w / 2, h / 2
        img = ((X - CenterX) ** 2 + (Y - CenterY) ** 2)
        return img

    def centerCrop(img, w, h):
        # crops around center
        maxY, maxX = img.shape[0], img.shape[1]
        x1, y1 = int(maxX / 2 - w / 2), int(maxY / 2 - h / 2)
        x2, y2 = int(maxX / 2 + w / 2), int(maxY / 2 + h / 2)
        crop = img[y1:y2, x1:x2]
        return crop

    def rescale_frame(res, percent=75):
        # rescales image
        width = int(res.shape[1] * percent / 100)
        height = int(res.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(res, dim, interpolation=cv2.INTER_AREA)