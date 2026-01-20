import cv2, torch, numpy as np
from AI.ChessCNN import ChessCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from AI.Ellipse_crop import EllipseCrop
from image_processing import binary_treshold_finder

val_tf = A.Compose([
    A.Resize(50, 50),
    A.Normalize(mean=(0.5,), std=(0.5,), p=1.0),
    ToTensorV2()
])

class BoardIdentification:
    def __init__(self, frame, board_model, device=None):
        self.frame = frame
        self.model = board_model
        self.device = device

        self.pieces = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]
        self.colors = ["W", "B"]

    def preprocess(self, square):
        if len(square.shape) == 3 and square.shape[2] == 3:
            square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        # ToTensorV2 zamieni to na (1, H, W).
        img = val_tf(image=square)['image'].unsqueeze(0)

        return img.to(self.device)

    def detect_color(self, sq, threshold):
        h, w = sq.shape[:2]

        ch, cw = h // 2, w // 2
        roi = sq[ch - 8:ch + 8, cw - 8:cw + 8]

        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        mean_val = roi_gray.mean()

        return "W" if mean_val > threshold else "B"

    @staticmethod
    def encode_board(board_2d: list[list[str]]) -> list[list[str]]:
        mapping = {
            'King': 'K',
            'Queen': 'Q',
            'Rook': 'R',
            'Bishop': 'B',
            'Knight': 'N',
            'Pawn': 'P'
        }

        encoded_board = []
        for row in board_2d:
            encoded_row = []
            for piece_str in row:
                if piece_str == ' ' or piece_str is None:
                    encoded_row.append(' ')
                    continue
                color, name = piece_str.split('_', 1)
                letter = mapping.get(name, ' ')
                if color == 'B':
                    letter = letter.lower()
                encoded_row.append(letter)
            encoded_board.append(encoded_row)

        return encoded_board

    def identify(self, size=8):
        #number = 0
        h, w = self.frame.shape[:2]
        sh, sw = h // size, w // size
        board = []
        with torch.no_grad():
            for r in range(size):
                row = []
                for c in range(size):
                    sq = self.frame[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                    thr, bright_node, dark_node = binary_treshold_finder.get_local_threshold(self.frame, r, c)


                    ellipse_crop = EllipseCrop()
                    if ellipse_crop.find(sq, thr, bright_node, dark_node):
                        # if number == 0:
                        #     sq = ellipse_crop.apply(sq, thr, bright_node, dark_node, step_visualize=True)
                        #     number = 1
                        # else:
                        sq = ellipse_crop.apply(sq, thr, bright_node, dark_node)
                        _, sq = cv2.threshold(sq, thr, 255, cv2.THRESH_BINARY)
                        color = self.detect_color(sq, thr)
                        # cv2.imshow("board", sq)
                        # cv2.waitKey(500)
                        img = self.preprocess(sq)
                        pp = self.model(img)
                        pi = pp.argmax(1).item()

                        row.append(f"{color}_{self.pieces[pi]}")

                    else:
                        row.append(' ')
                board.append(row)

        # for a in board:
        #     print(a)
        return self.encode_board(board)