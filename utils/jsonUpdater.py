import json
import os
import chess

class JsonUpdater:
    def __init__(self, filename="data.json", capacity=3):
        self.filename = filename
        self.capacity = capacity
        self.data = []
        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def get_data(self):
        self.data = self._load()
        return self.data

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

        if len(self.data) > 0:
            last_board = chess.Board(self.data[-1])
            print(last_board)
            if str(fen).split()[0] == str(self.data[-1]).split()[0]:
                return
            if not self.is_transition_possible(self.data[-1], fen):
                raise ValueError("Niemożliwy ruch!")

        self.data.append(fen)

        if len(self.data) > self.capacity:
            self.data.pop(0)

        self._save()

    def convert_chessBoard(self, newlist):
        if len(newlist) != 8 or any(len(row) != 8 for row in newlist):
            raise ValueError("newlist must be 8x8")

        board = chess.Board.empty()

        piece_map = {
            'P': chess.Piece(chess.PAWN, chess.WHITE),
            'R': chess.Piece(chess.ROOK, chess.WHITE),
            'N': chess.Piece(chess.KNIGHT, chess.WHITE),
            'B': chess.Piece(chess.BISHOP, chess.WHITE),
            'Q': chess.Piece(chess.QUEEN, chess.WHITE),
            'K': chess.Piece(chess.KING, chess.WHITE),

            'p': chess.Piece(chess.PAWN, chess.BLACK),
            'r': chess.Piece(chess.ROOK, chess.BLACK),
            'n': chess.Piece(chess.KNIGHT, chess.BLACK),
            'b': chess.Piece(chess.BISHOP, chess.BLACK),
            'q': chess.Piece(chess.QUEEN, chess.BLACK),
            'k': chess.Piece(chess.KING, chess.BLACK),
        }

        for row in range(8):
            for col in range(8):
                cell = newlist[row][col]

                if cell == ' ':
                    continue

                try:
                    piece = piece_map[cell]
                except Exception:
                    raise ValueError(f"Invalid cell value: {cell}")

                square = chess.square(col, 7 - row)
                board.set_piece_at(square, piece)

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
