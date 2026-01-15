import json
import os
import chess
from utils.logger import Logger

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

    def add(self, newlist, compare=None):
        camera_board = self.convert_chessBoard(newlist)
        camera_fen_only = camera_board.board_fen()

        self.data = self._load()

        final_board = camera_board

        if len(self.data) > 0 and compare:
            last_fen = self.data[-1]
            validation_board = chess.Board(last_fen)

            move = chess.Move.from_uci(compare)
            if move in validation_board.legal_moves:
                validation_board.push(move)

                if validation_board.board_fen() != camera_fen_only:
                    raise ValueError(f"Rozbieżność! Ruch {compare} nie zgadza się z widokiem kamery.")

                final_board = validation_board
            else:
                raise ValueError(f"Ruch {compare} jest nielegalny w tej pozycji!")

        if len(self.data) > 0 and not compare:
            if camera_fen_only == chess.Board(self.data[-1]).board_fen():
                raise ValueError("Brak zmian na planszy - nie dodano do pliku.")


        if final_board.is_checkmate():
            winner = "Black" if final_board.turn == chess.WHITE else "White"
            error_msg = f"MAT! Wygrywa: {winner}"
            Logger.log(error_msg)

            raise Exception(error_msg)

        # 5. Zapisywanie poprawnego stanu
        final_fen = final_board.fen()
        self.data.append(final_fen)

        if len(self.data) > self.capacity:
            self.data.pop(0)

        self._save()
        Logger.log(f"Dodano FEN: {final_fen}")
        print("Pomyślnie zaktualizowano stan szachownicy.")

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
