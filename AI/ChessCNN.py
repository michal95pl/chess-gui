import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):
    def __init__(self, num_pieces: int = 7, num_colors: int = 3):
        super().__init__()
        # 3×100×100  --> 128×1×1
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 50×50
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 25×25
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)                        # 128×1×1
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc_piece = nn.Linear(128, num_pieces)
        self.fc_color = nn.Linear(128, num_colors)

    def forward(self, x):
        z = self.features(x)
        z = self.fc(z)
        return self.fc_piece(z), self.fc_color(z)