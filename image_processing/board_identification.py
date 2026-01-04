import cv2, torch
from AI.ChessCNN import ChessCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from AI.Ellipse_crop import EllipseCrop
from image_processing import binary_treshold_finder
import numpy as np

val_tf = A.Compose([
    A.Resize(30, 30),
    A.ShiftScaleRotate(
        rotate_limit=10,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.4
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
    ToTensorV2()
])

class BoardIdentification:
    def __init__(self, frame, model_path="AI/chess_best.pth", device=None):
        self.frame = frame
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN(num_pieces=6)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.pieces = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]
        self.colors = ["W", "B"]

    def preprocess(self, square):
        if len(square.shape) == 3 and square.shape[2] == 3:
            square = square[:, :, ::-1]  # BGR→RGB
        else:
            square = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
        square = cv2.resize(square, (100, 100))
        img = val_tf(image=square)['image'].unsqueeze(0)
        return img.to(self.device)

    def detect_color(self, sq, threshold=128):
        h, w = sq.shape[:2]

        ch, cw = h // 2, w // 2
        roi = sq[ch - 1:ch + 1, cw - 1:cw + 1]

        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        mean_val = roi_gray.mean()

        return "W" if mean_val > threshold else "B"

    @staticmethod
    def encode(piece_str: str) -> str:
        color, name = piece_str.split('_', 1)
        mapping = {
            'King': 'K',
            'Queen': 'Q',
            'Rook': 'R',
            'Bishop': 'B',
            'Knight': 'N',
            'Pawn': 'P'
        }

        letter = mapping.get(name, ' ')  # jeśli nieznana figura → puste
        if color == 'B':
            letter = letter.lower()  # czarne figury w małych literach

        return letter

    def identify(self, size=8):
        h, w = self.frame.shape[:2]
        sh, sw = h // size, w // size
        board = []
        with torch.no_grad():
            for r in range(size):
                row = []
                for c in range(size):
                    sq = self.frame[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                    thr, bright_node, dark_node = binary_treshold_finder.get_local_threshold(self.frame, r, c)
                    _, sq = cv2.threshold(sq, thr, 255, cv2.THRESH_BINARY)

                    ellipse_crop = EllipseCrop()
                    if ellipse_crop.find(sq, thr, bright_node, dark_node):
                        sq = ellipse_crop.apply(sq)
                        color = self.detect_color(sq)

                        img = self.preprocess(sq)
                        pp = self.model(img)
                        pi = pp.argmax(1).item()

                        row.append(self.encode(f"{color}_{self.pieces[pi]}"))

                    else:
                        row.append(' ')
                board.append(row)

        for a in board:
            print(a)
        return board