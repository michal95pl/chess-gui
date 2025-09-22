import tkinter as tk

from GUI.cameraSelectPage import CameraSelectPage
from GUI.colorCalibrationPage import ColorCalibrationPage
from GUI.gamePage import GamePage
from image_processing.board_transformation import BoardTransformation
from time import sleep
from utils.logger import Logger

Logger.reset_log()
Logger.log("Chess GUI started.")

root = tk.Tk()
root.title("Chess GUI")
root.iconphoto(False, tk.PhotoImage(file="assets/icon.png"))
root.geometry("900x600")

def color_calibration_page_listener(camera_index: int, green: tuple, red: tuple):
    sleep(2) # todo: add sync between camera threads
    GamePage(root, camera_index, BoardTransformation(green, red, root))

def camera_select_page_listener(camera_index: int):
    ColorCalibrationPage(root, camera_index, color_calibration_page_listener)

root.resizable(True, True)

CameraSelectPage(root, camera_select_page_listener)

root.mainloop()