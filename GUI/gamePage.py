from GUI.videoFrame import VideoFrame
import tkinter as tk
from image_processing.board_transformation import BoardTransformation
from communication.communication import Communication
from GUI.ChessGUI import ChessGUI

class GamePage:

    def __init__(self, root: tk.Tk, camera_index: int, board_transformation: BoardTransformation, communication: Communication):
        self.root = root

        # ramki dla rozdzielenia widoków
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # widok kamery
        self.video_frame = VideoFrame(
            self.left_frame,
            camera_index,
            video_size=(800, 800),
            board_transformation=board_transformation,
            communication=communication,
            camera_delay=True
        )
        self.video_frame.start()

        # GUI szachownicy
        self.chess_gui = ChessGUI(
            self.right_frame,
            self.video_frame.get_board_state
        )
