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

    def get_move_made(self, last_fen, current_pieces_fen):
        board = chess.Board(last_fen)
        for move in board.legal_moves:
            board.push(move)
            if board.board_fen() == current_pieces_fen:
                board.pop()
                return move
            board.pop()
        return None

    def add(self, newlist, compare=None):
        camera_board = self.convert_chessBoard(newlist)
        camera_fen_only = camera_board.board_fen()
        self.data = self._load()

        final_board = camera_board

        # PRZYPADEK 1: Mamy ruch z serwera
        if len(self.data) > 0 and compare:
            compare = self.flip_uci_move(compare) #trzeba dopracować
            last_fen = self.data[-1]
            validation_board = chess.Board(last_fen)

            move = chess.Move.from_uci(compare)
            if move in validation_board.legal_moves:
                validation_board.push(move)

                if validation_board.board_fen() != camera_fen_only:
                    raise ValueError(f"Rozbieżność! Serwer: {compare}, Kamera widzi coś innego.")

                final_board = validation_board
            else:
                raise ValueError(f"Ruch {compare} jest nielegalny w tej pozycji!")

                # PRZYPADEK 2: Gracz wykonał ruch
        elif len(self.data) > 0 and not compare:
            last_fen = self.data[-1]
            old_board = chess.Board(last_fen)  # Ładujemy stary stan (z dobrą turą)

            # 1. Sprawdź czy cokolwiek się zmieniło
            if camera_fen_only == old_board.board_fen():
                raise ValueError("Brak zmian na planszy.")

            # 2. Znajdź jaki to był ruch
            move = self.get_move_made(last_fen, camera_fen_only)

            if move:
                old_board.push(move)  # To automatycznie zmieni 'w' na 'b' w FEN!
                final_board = old_board
                Logger.log(f"Wykonano ruch: {move}")
            else:
                raise ValueError("Wykryto nielegalny ruch lub błąd rozpoznawania figury!")

            move = self.get_move_made(last_fen, camera_board.fen())
            Logger.log(f"Wykonano ruch: {move}")
            print(f"Wykonano ruch: {move}")

        # Sprawdzanie matu
        if final_board.is_checkmate():
            winner = "Black" if final_board.turn == chess.WHITE else "White"
            error_msg = f"MAT! Wygrywa: {winner}"
            Logger.log(error_msg)
            raise Exception(error_msg)

        # Zapis
        final_fen = final_board.fen()
        self.data.append(final_fen)
        if len(self.data) > self.capacity:
            self.data.pop(0)

        self._save()
        Logger.log(f"Dodano FEN: {final_fen}")
        print("Pomyślnie zaktualizowano stan szachownicy.")
        for row in newlist:
            print(row)

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

    def flip_uci_move(self, uci_move: str) -> str:
        if not uci_move or len(uci_move) < 4:
            return uci_move

        files = 'abcdefgh'
        ranks = '12345678'

        result = ""
        # Przetwarzamy parami (pole startowe i pole docelowe)
        for i in range(0, 4, 2):
            f = uci_move[i]  # litera (kolumna)
            r = uci_move[i + 1]  # cyfra (rząd)

            # Lustrzane odbicie litery: a(0) -> h(7), b(1) -> g(6) itd.
            new_f = files[7 - files.index(f)]
            # Lustrzane odbicie cyfry: 1 -> 8, 2 -> 7 itd.
            new_r = ranks[7 - ranks.index(r)]

            result += new_f + new_r
            print(result)
        return result