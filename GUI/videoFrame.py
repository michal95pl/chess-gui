import os
from threading import Thread, Lock
from PIL import Image, ImageTk
import tkinter as tk
import cv2
from image_processing.board_transformation import BoardTransformation
from utils.logger import Logger

class VideoFrame(Thread):

    def __init__(self, root, camera_index: int, video_size=(500, 500), board_transformation: BoardTransformation = None):
        super().__init__(daemon=True, name="VideoFrameThread")

        self.camera_index = camera_index
        self.root = root
        self.update_video = False
        self.actual_frame = None
        self.board_transformation = board_transformation
        self.video_size = video_size
        self.test_image_path = "assets/board_test.png"

        # create blank image for initialization
        blank_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        self.video_label = tk.Label(root, image=ImageTk.PhotoImage(blank_image))
        self.video_label.image = blank_image
        self.video_label.pack()

    def show_frame(self, frame):
        tk_frame = self.__convert_frame_to_tk(frame)
        self.video_label.configure(image=tk_frame)
        self.video_label.image = tk_frame

    def start(self):
        self.update_video = True
        super().start()

    def stop(self):
        self.update_video = False

    def run(self):
        if self.camera_index == -1:
            if not os.path.exists(self.test_image_path):
                Logger.log("Error: {test_image_path} not found.")
                print("Error: {test_image_path} not found.")
                return
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                Logger.log("Error: Could not read {test_image_path}.")
                print("Error: Could not read {test_image_path}.")
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.video_size)  # todo: change size
            if self.board_transformation is not None:
                frame = self.board_transformation.transform(frame)
            self.actual_frame = frame
            self.show_frame(frame)

        else:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                Logger.log("Error: Could not open video.")
                print("Error: Could not open video.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    Logger.log("Error: Could not read frame.")
                    print("Error: Could not read frame.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.video_size) # todo: change size
                if self.board_transformation is not None:
                    frame = self.board_transformation.transform(frame)
                self.actual_frame = frame

                # it should be before show_frame function to make sure that thread stops immediately before showing new frame
                if not self.update_video:
                    break
                self.show_frame(frame)
            cap.release()

    def __convert_frame_to_tk(self, frame):
        frame = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=frame)

