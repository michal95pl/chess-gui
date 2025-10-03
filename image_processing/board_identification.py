import cv2, torch, os, numpy as np
from AI.ChessCNN import ChessCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

val_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class BoardIdentification:
    def __init__(self, frame, model_path="AI/chess_best.pth", device=None):
        self.frame = frame
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN(num_pieces=7, num_colors=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.pieces = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King", "Empty"]
        self.colors = ["W", "B", "None"]

    def preprocess(self, square):
        if len(square.shape) == 3 and square.shape[2] == 3:
            square = square[:, :, ::-1]  # BGR→RGB
        else:
            square = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
        square = cv2.resize(square, (100, 100))
        img = val_tf(image=square)['image'].unsqueeze(0)
        return img.to(self.device)

    def identify(self, size=8):
        h, w = self.frame.shape[:2]
        sh, sw = h // size, w // size
        board = []
        with torch.no_grad():
            for r in range(size):
                row = []
                for c in range(size):
                    sq = self.frame[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                    img = self.preprocess(sq)
                    pp, pc = self.model(img)
                    pi, ci = pp.argmax(1).item(), pc.argmax(1).item()
                    row.append(f"{self.colors[ci]}_{self.pieces[pi]}")
                board.append(row)

        for a in board:
            print(a)
        return board