import os
import torch
from threading import Thread, Lock
from time import sleep

from PIL import Image, ImageTk
import tkinter as tk
import cv2

from AI.ChessCNN import ChessCNN
from image_processing.board_identification import BoardIdentification
from image_processing.board_transformation import BoardTransformation
from utils.logger import Logger
from utils.jsonUpdater import JsonUpdater
from communication.communication import Communication
from GUI.ChessGUI import ChessGUI

class VideoFrame(Thread):

    def __init__(self, root, camera_index: int, video_size=(500, 500), board_transformation: BoardTransformation = None, communication: Communication = None):
        super().__init__(daemon=True, name="VideoFrameThread")

        self.camera_index = camera_index
        self.root = root
        self.update_video = False
        self.actual_frame = None
        self.board_transformation = board_transformation
        self.communication = communication
        self.video_size = video_size
        self.test_image_path = "assets/ChessBoard_1.png"
        self.identified_pieces = None
        self.jsonUpdater = JsonUpdater()
        self.chess_gui = None
        self.model = ChessCNN(num_pieces=6)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load("AI/chess_best.pth", map_location=self.device))
        self.model.to(self.device).eval()
        self.start_flag = False
        self.root.bind("<Return>", self._on_enter)
        self.turn = 0

        # create blank image for initialization
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.LEFT)

        blank_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        self.video_label = tk.Label(root, image=ImageTk.PhotoImage(blank_image))
        self.video_label.pack()

    def _on_enter(self, event=None):
        if not self.start_flag:
            Logger.log("START – analiza ruchów aktywna")
            self.start_flag = True

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
                return
            frame = cv2.imread(self.test_image_path)
            if frame is None:
                Logger.log("Error: Could not read {test_image_path}.")
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.video_size)
            try:
                frame = self.board_transformation.transform(frame)
                self.identified_pieces = BoardIdentification(frame, self.model, self.device).identify()
                if self.start_flag:
                    self.jsonUpdater.add(self.identified_pieces)
                    Logger.log("Successfully added board to json file.")
                    # if self.check_turn():
                    #     self.communication.send({
                    #         'command': 'get_move', 'boards': self.jsonUpdater.get_data()
                    #     })

                if self.chess_gui is None:
                    self.chess_gui = ChessGUI(self.root, self.get_board_state())
            except Exception as e:
                print(e)
                Logger.log(e.__str__())
            self.actual_frame = frame
            self.show_frame(frame)

        else:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                Logger.log("Error: Could not open video.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    Logger.log("Error: Could not read frame.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.video_size)
                try:
                    frame = self.board_transformation.transform(frame)
                    self.identified_pieces = BoardIdentification(frame, self.model, self.device).identify()
                    if self.start_flag:
                        self.jsonUpdater.add(self.identified_pieces)
                        if self.check_turn():
                            self.communication.send({
                                'command': 'get_move', 'boards': self.jsonUpdater.get_data()
                            })
                    if self.chess_gui is None:
                        self.chess_gui = ChessGUI(self.root, self.get_board_state())
                except Exception as e:
                    print(e)
                    Logger.log(e.__str__())
                self.actual_frame = frame

                if not self.update_video:
                    break
                self.show_frame(frame)
                sleep(2)
            cap.release()

    def __convert_frame_to_tk(self, frame):
        frame = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=frame)

    def get_board_state(self):
        if self.identified_pieces:
            return self.identified_pieces
        return [[' '] * 8 for _ in range(8)]

    def check_turn(self):
        answear = self.turn % 2 == 0
        print("White's turn." if answear else "Black's turn.")
        self.turn += 1
        return answear

