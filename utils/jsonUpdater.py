import json
import os
import chess

class JsonUpdater:
    def __init__(self, filename="data.json", capacity=3):
        self.filename = filename
        self.capacity = capacity
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.filename):
            return []
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

    def add(self, newlist):
        board = self.convert_chessBoard(newlist)
        fen = board.fen()

        self.data = self._load()

        if self.data:
            last_board = chess.Board(self.data[-1])
            print(last_board)
            if str(fen).split()[0] == str(self.data[-1]).split()[0]:
                return
            if not self.is_transition_possible(self.data[-1], fen):
                raise ValueError("Niemożliwy ruch!")

        # Dodanie nowego FEN
        self.data.append(fen)

        if len(self.data) > self.capacity:
            self.data.pop(0)

        self._save()

    def convert_chessBoard(self, newlist):
        if len(newlist) != 8 or any(len(row) != 8 for row in newlist):
            raise ValueError("newlist must be 8x8")

        board = chess.Board.empty()

        piece_map = {
            "Pawn": chess.PAWN,
            "Rook": chess.ROOK,
            "Knight": chess.KNIGHT,
            "Bishop": chess.BISHOP,
            "Queen": chess.QUEEN,
            "King": chess.KING,
        }

        color_map = {
            "W": chess.WHITE,
            "B": chess.BLACK
        }

        for row in range(8):
            for col in range(8):
                cell = newlist[row][col]

                if cell == "_":
                    continue

                try:
                    color_char, piece_name = cell.split("_")
                    color = color_map[color_char]
                    piece_type = piece_map[piece_name]
                except Exception:
                    raise ValueError(f"Invalid cell value: {cell}")

                square = chess.square(col, 7 - row)
                board.set_piece_at(square, chess.Piece(piece_type, color))

        return board

    def is_transition_possible(self, fen_from: str, fen_to: str) -> bool:
        board_from = chess.Board(fen_from)
        board_to = chess.Board(fen_to)

        for move in board_from.legal_moves:
            board_from.push(move)
            if board_from.board_fen() == board_to.board_fen():
                board_from.pop()
                return True
            board_from.pop()
        return False
