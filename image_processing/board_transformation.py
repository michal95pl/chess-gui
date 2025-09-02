import cv2
import numpy as np

class BoardTransformation:

    def __rgb2hsv(self, rgb: tuple):
        color = np.uint8([[list(rgb)]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        return tuple(hsv_color[0][0])

    def __init__(self, green_calibration: tuple = None, red_calibration: tuple = None):
        self.green_hsv_calibration = self.__rgb2hsv(green_calibration)
        self.red_hsv_calibration = self.__rgb2hsv(red_calibration)

    def transform(self, frame):

        return frame
