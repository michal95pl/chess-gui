import os
import tkinter as tk
import cv2

class CameraSelectPage:

    @staticmethod
    def __get_available_cameras(max_camera_index=3):
        cam = [-1]
        for i in range(max_camera_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cam.append(i)
                cap.release()
        return cam

    def __on_button_click(self):
        if self.dropdown_value.get() != "Select Camera":
            while self.root.winfo_children():
                self.root.winfo_children()[0].destroy()
            self.listener(int(self.dropdown_value.get()))

    def __init__(self, root, listener: callable):
        self.root = root
        self.listener = listener
        self.dropdown_value = tk.StringVar(root, value="Select Camera")
        dropdown = tk.OptionMenu(root, self.dropdown_value, *CameraSelectPage.__get_available_cameras())
        button = tk.Button(root, text="Connect", command=self.__on_button_click)

        dropdown.pack()
        button.pack()