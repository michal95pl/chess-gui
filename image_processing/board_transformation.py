import cv2
import numpy as np
import tkinter as tk

from utils.logger import Logger


class BoardTransformation:

    def __get_color_centers(self, mask, frame, color=(0, 255, 0), min_area=50):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 8, color, 2)
        return centers


























    def __rgb2hsv(self, rgb: tuple):
        color = np.uint8([[list(rgb)]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        return tuple(hsv_color[0][0])

    def __init__(self, green_calibration: tuple, red_calibration: tuple, root):
        self.green_hsv_calibration = self.__rgb2hsv(green_calibration)
        self.red_hsv_calibration = self.__rgb2hsv(red_calibration)
        self.root = root

    def __get_max_contour_area(contours):
        imax = 0
        max_area = 0
        for i in range(len(contours)):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                imax = i
                max_area = area
        return contours[imax], max_area

    def transform(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # mask colors
        green_upper = np.array([self.green_hsv_calibration[0]*1.4, self.green_hsv_calibration[1]*1.4, self.green_hsv_calibration[2]*1.4])
        green_lower = np.array([self.green_hsv_calibration[0]*0.6, self.green_hsv_calibration[1]*0.6, self.green_hsv_calibration[2]*0.6])
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        red_upper = np.array([self.red_hsv_calibration[0]*1.4, self.red_hsv_calibration[1]*1.4, self.red_hsv_calibration[2]*1.4])
        red_lower = np.array([self.red_hsv_calibration[0]*0.6, self.red_hsv_calibration[1]*0.6, self.red_hsv_calibration[2]*0.6])
        red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        green_centers = self.__get_color_centers(green_mask, frame, color=(0, 255, 0))
        red_centers = self.__get_color_centers(red_mask, frame, color=(0, 0, 255))

        if green_centers is None or red_centers is None:
            print("Error: Green and Red centers are not found.")
            Logger.log("Error: Green and Red centers are not found.")
        else:
            print("Green and Red centers are found.")
            Logger.log("Green and Red centers are found.")

        lewy_gorny_rog = red_centers[0]
        


        return frame
