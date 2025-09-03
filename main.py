import tkinter as tk

import cv2

from GUI.cameraSelectPage import CameraSelectPage
from GUI.colorCalibrationPage import ColorCalibrationPage
from GUI.gamePage import GamePage
from image_processing.board_transformation import BoardTransformation
from time import sleep


root = tk.Tk()
root.title("Chess GUI")
root.iconphoto(False, tk.PhotoImage(file="assets/icon.png"))
root.geometry("1920x1080")

def color_calibration_page_listener(camera_index: int, green: tuple, red: tuple):
    sleep(2) # todo: add sync between camera threads
    GamePage(root, camera_index, BoardTransformation(green, red, root))

def camera_select_page_listener(camera_index: int):
    ColorCalibrationPage(root, camera_index, color_calibration_page_listener)

cv_img = cv2.imread('assets/board_test.png')
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
cv_img = cv2.resize(cv_img, (1000, 600))

root.resizable(False, False)

CameraSelectPage(root, camera_select_page_listener)

root.mainloop()