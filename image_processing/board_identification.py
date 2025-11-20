import cv2, torch
from AI.ChessCNN import ChessCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from AI.Ellipse_crop import EllipseCrop

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

    def identify(self, size=8):
        h, w = self.frame.shape[:2]
        sh, sw = h // size, w // size
        board = []
        with torch.no_grad():
            for r in range(size):
                row = []
                for c in range(size):
                    sq = self.frame[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                    _, sq = cv2.threshold(sq, 80, 255, cv2.THRESH_BINARY)
                    ellipse_crop = EllipseCrop()
                    if ellipse_crop.find(sq):
                        sq = ellipse_crop.apply(sq)
                        img = self.preprocess(sq)
                        pp = self.model(img)
                        pi = pp.argmax(1).item()
                        row.append(f"{self.pieces[pi]}")
                    else:
                        row.append("_")
                board.append(row)

        for a in board:
            print(a)
        return board