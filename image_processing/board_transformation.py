import cv2
import numpy as np
import tkinter as tk

class BoardTransformation:

    def __rgb2hsv(self, rgb: tuple):
        color = np.uint8([[list(rgb)]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        return tuple(hsv_color[0][0])

    def __init__(self, green_calibration: tuple, red_calibration: tuple, root):
        self.green_hsv_calibration = self.__rgb2hsv(green_calibration)
        self.red_hsv_calibration = self.__rgb2hsv(red_calibration)
        self.root = root

        self.a_val = tk.IntVar()
        self.b_val = tk.IntVar()
        self.c_val = tk.IntVar()
        self.d_val = tk.IntVar()
        self.e_val = tk.IntVar()
        self.f_val = tk.IntVar()
        #
        # tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="a", variable=self.a_val).pack()
        # tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, label="b", variable=self.b_val).pack()
        # tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="c", variable=self.c_val).pack()
        # tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="d", variable=self.d_val).pack()
        # tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="e", variable=self.e_val).pack()
        # tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="f", variable=self.f_val).pack()

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

    def __get_center_of_circles(self, frame):
        frame = cv2.medianBlur(frame, 5)
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=15, minDist=80, param1=20, param2=20, minRadius=3,
                                   maxRadius=22)

        return circles[0, :, 0:2] if circles is not None else None

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


        green_centers = self.__get_center_of_circles(green_mask)
        red_centers = self.__get_center_of_circles(red_mask)

        if green_centers is None or red_centers is None or len(green_centers) != 2 or len(red_centers) != 1:
            print("Error: Could not find board.") #todo: add logs
        else:
            left_upper_point = red_centers[0]

            print("x: ", red_centers[0][1] - green_centers[0][1])
            print("y: ", red_centers[0][1] - green_centers[1][1])
            print()

            if green_centers[0][0] < green_centers[0][1]:

                if red_centers[0][1] - green_centers[0][1] < 0 or red_centers[0][1] - green_centers[1][1] < 0:
                    left_lower_point = green_centers[0]
                    right_lower_point = green_centers[1]
                else:
                    left_lower_point = green_centers[1]
                    right_lower_point = green_centers[0]
            else:
                if red_centers[0][1] - green_centers[0][1] < 0 or red_centers[0][1] - green_centers[1][1] < 0:
                    left_lower_point = green_centers[1]
                    right_lower_point = green_centers[0]
                else:
                    left_lower_point = green_centers[0]
                    right_lower_point = green_centers[1]

            cv2.circle(frame, (int(right_lower_point[0]), int(right_lower_point[1])), 10, (0, 255, 0), 2)
            cv2.circle(frame, (int(left_lower_point[0]), int(left_lower_point[1])), 10, (0, 0, 255), 2)
            cv2.circle(frame, (int(left_upper_point[0]), int(left_upper_point[1])), 10, (255, 0, 0), 2)

        return frame
