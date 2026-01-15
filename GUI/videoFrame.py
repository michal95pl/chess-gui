import os
import time

import torch
from threading import Thread, Lock
from time import sleep

from PIL import Image, ImageTk
import tkinter as tk
import cv2
from pyexpat.errors import messages

from AI.ChessCNN import ChessCNN
from image_processing.board_identification import BoardIdentification
from image_processing.board_transformation import BoardTransformation
from utils.logger import Logger
from utils.jsonUpdater import JsonUpdater
from communication.communication import Communication

class VideoFrame(Thread):

    def __init__(self, root, camera_index: int, video_size=(500, 500), board_transformation: BoardTransformation = None, communication: Communication = None, camera_delay=False):
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
        self.model = ChessCNN(num_pieces=6)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load("AI/chess_best.pth", map_location=self.device))
        self.model.to(self.device).eval()
        self.start_flag = False
        self.root.bind("<Return>", self._on_enter)
        self.root.focus_force()
        self.turn = 0
        self.camera_delay = camera_delay

        # create blank image for initialization
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.LEFT)

        blank_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        self.video_label = tk.Label(root, image=ImageTk.PhotoImage(blank_image))
        self.video_label.pack()

    def _on_enter(self, event=None):
        if not self.start_flag:
            Logger.log("START – analiza ruchów aktywna")
            print("START – analiza ruchów aktywna")
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
        message = None
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
            if self.board_transformation is not None:
                frame = self.board_transformation.transform(frame)
                self.identified_pieces = BoardIdentification(frame, self.model, self.device).identify()
                self.jsonUpdater.add(self.identified_pieces)

            self.actual_frame = frame
            self.show_frame(frame)

        else:
            cap = cv2.VideoCapture("http://192.168.220.108:8080/video")
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
                    if self.board_transformation is not None:
                        frame = self.board_transformation.transform(frame)
                        self.identified_pieces = BoardIdentification(frame, self.model, self.device).identify()
                        if self.start_flag:
                            if message == None:
                                self.jsonUpdater.add(self.identified_pieces)
                            else:
                                self.jsonUpdater.add(self.identified_pieces, compare=message)
                                message = None
                            if self.check_turn() == False:
                                self.communication.send({
                                    'command': 'get_move', 'boards': self.jsonUpdater.get_data()
                                })

                                start = time.time()
                                while True:
                                    raw_msg = self.communication.get_message()
                                    if raw_msg is not None:
                                        if isinstance(raw_msg, dict):
                                            move_to_apply = raw_msg.get('move')
                                        else:
                                            move_to_apply = str(raw_msg).split("move:")[-1].strip()

                                        print(f"Wyodrębniony ruch: {move_to_apply}")
                                        message = move_to_apply  # To zostanie użyte w następnej iteracji self.jsonUpdater.add
                                        break

                                    if time.time() - start > 15:  # Zwiększyłem do 15 zgodnie z Twoim Loggerem
                                        Logger.log("Błąd: Serwer nie odpowiedział w ciągu 15 sekund.")
                                        print("Błąd: Timeout komunikacji.")
                                        message = None  # Czyścimy, żeby nie dodać śmieci przy następnej klatce
                                        break

                                    sleep(0.1)

                except Exception as e:
                    print(e)
                    Logger.log(e.__str__())
                self.actual_frame = frame

                if not self.update_video:
                    break
                self.show_frame(frame)
                if self.camera_delay:
                    sleep(1)
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

