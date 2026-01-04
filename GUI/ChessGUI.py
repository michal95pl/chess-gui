import tkinter as tk
from PIL import Image, ImageTk

class ChessGUI:
    SQUARE_SIZE = 60

    def __init__(self, parent, board_state_provider):
        """
        parent – tk.Frame / tk.Tk
        board_state_provider – callable returning 8x8 board (chars)
        """
        self.parent = parent
        self.board_state_provider = board_state_provider

        self.canvas = tk.Canvas(
            parent,
            width=self.SQUARE_SIZE * 8,
            height=self.SQUARE_SIZE * 8,
            bg="#222"
        )
        self.canvas.pack(side=tk.RIGHT, padx=10)

        self.images = {}
        self._load_images()
        self._draw_board()
        self.update()

    def _load_images(self):
        mapping = {
            'r': 'black_rook.png', 'n': 'black_knight.png', 'b': 'black_bishop.png',
            'q': 'black_queen.png', 'k': 'black_king.png', 'p': 'black_pawn.png',
            'R': 'white_rook.png', 'N': 'white_knight.png', 'B': 'white_bishop.png',
            'Q': 'white_queen.png', 'K': 'white_king.png', 'P': 'white_pawn.png',
        }
        for k, v in mapping.items():
            img = Image.open(f"./assets/chessgui/{v}").resize((45, 45))
            self.images[k] = ImageTk.PhotoImage(img)

    def _draw_board(self):
        self.canvas.delete("square")
        colors = ("#345", "#567")
        for y in range(8):
            for x in range(8):
                color = colors[(x + y) % 2]
                self.canvas.create_rectangle(
                    x * self.SQUARE_SIZE,
                    y * self.SQUARE_SIZE,
                    (x + 1) * self.SQUARE_SIZE,
                    (y + 1) * self.SQUARE_SIZE,
                    fill=color,
                    outline=color,
                    tags="square"
                )

    def _draw_pieces(self, board):
        self.canvas.delete("piece")
        for y in range(8):
            for x in range(8):
                p = board[y][x]
                if p != ' ':
                    self.canvas.create_image(
                        x * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                        y * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                        image=self.images[p],
                        tags="piece"
                    )

    def update(self):
        board = self.board_state_provider()
        if board is not None:
            self._draw_board()
            self._draw_pieces(board)
        self.parent.after(100, self.update)
