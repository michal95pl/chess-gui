import cv2, torch
from AI.ChessCNN import ChessCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from AI.Ellipse_crop import EllipseCrop
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

    import cv2
    import numpy as np

    def get_local_threshold(self, row, col, blocks=4):
        h, w = self.frame.shape[:2]

        # rozmiar bloku 4x4
        bh, bw = h // blocks, w // blocks

        # ustalenie bloku (dla 8x8: 2 pola = 1 blok)
        br = row // (8 // blocks)
        bc = col // (8 // blocks)

        # wycinamy blok
        block = self.frame[br * bh:(br + 1) * bh, bc * bw:(bc + 1) * bw]

        # grayscale
        if len(block.shape) == 3:
            gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        else:
            gray = block

        # histogram 0..255
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # dominujący ciemny odcień (0–127)
        dark_mode = np.argmax(hist[1:128])
        # dominujący jasny odcień (128–255)
        bright_mode = np.argmax(hist[128:]) + 128

        # threshold: środek pomiędzy dwoma odcieniami
        thr = (dark_mode + bright_mode) / 2
        return thr

    def identify(self, size=8):
        h, w = self.frame.shape[:2]
        sh, sw = h // size, w // size
        board = []
        with torch.no_grad():
            for r in range(size):
                row = []
                for c in range(size):
                    sq = self.frame[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                    thr = self.get_local_threshold(r, c)
                    _, sq = cv2.threshold(sq, thr, 255, cv2.THRESH_BINARY)

                    ellipse_crop = EllipseCrop()
                    #Jest jakiś problem, wykrywa kółko tam gdzie go nie ma
                    if ellipse_crop.find(sq, thr):
                        sq = ellipse_crop.apply(sq)
                        color = self.detect_color(sq)

                        img = self.preprocess(sq)
                        pp = self.model(img)
                        pi = pp.argmax(1).item()

                        row.append(f"{color}_{self.pieces[pi]}")

                    else:
                        row.append("_")
                board.append(row)

        for a in board:
            print(a)
        return board