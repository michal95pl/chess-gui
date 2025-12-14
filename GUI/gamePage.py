from GUI.videoFrame import VideoFrame
import tkinter as tk
from image_processing.board_transformation import BoardTransformation

class GamePage(VideoFrame):

    def __init__(self, root: tk.Tk, camera_index: int, board_transformation: BoardTransformation):
        super().__init__(root, camera_index, (800, 600), board_transformation)
        self.start()