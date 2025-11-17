import os, cv2, torch, collections, albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from ChessCNN import ChessCNN


IMG_SIZE = 100
DATASET_DIR = "../assets/chess_pieces"
PIECES = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]
COLORS = ["W", "B"]

# ---------- transforms ----------
train_tf = A.Compose([
    A.Resize(120, 120),



    # A.ShiftScaleRotate(
    #     shift_limit=0.10,
    #     scale_limit=0.10,
    #     rotate_limit=30,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     p=0.6
    # ),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ---------- dataset ----------
class ChessDataset(Dataset):
    def __init__(self, root, pieces, colors, transform=None):
        self.samples, self.p2i, self.c2i = [], {p: i for i, p in enumerate(pieces)}, {c: i for i, c in enumerate(colors)}
        self.transform = transform
        for folder in os.listdir(root):
            path = os.path.join(root, folder)

            color, piece = folder.split("_")
            for fname in os.listdir(path):
                fnamepath = os.path.join(path, fname)

                if fname.lower().endswith(('.png')):
                    self.samples.append((fnamepath, piece, color))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, piece, color = self.samples[idx]
        img = cv2.imread(path)[:, :, ::-1]          # BGR → RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.transform: img = self.transform(image=img)['image']
        else: img = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1)
        return img, self.p2i[piece], self.c2i[color]

# ---------- helpers ----------
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    cp, cc, total = 0, 0, 0
    for x, yp, yc in loader:
        x, yp, yc = x.to(device), yp.to(device), yc.to(device)
        pp, pc = model(x)
        cp += (pp.argmax(1) == yp).sum().item()
        cc += (pc.argmax(1) == yc).sum().item()
        total += yp.size(0)
    return cp/total, cc/total

# ---------- main ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ChessDataset(DATASET_DIR, PIECES, COLORS, transform=train_tf)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds , batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # class weights for imbalance
    # ---------- build weights from original dataset ----------
    raw_indices = train_ds.indices
    piece_cnt = collections.Counter([ds.samples[i][1] for i in raw_indices])
    color_cnt = collections.Counter([ds.samples[i][2] for i in raw_indices])
    piece_weights = 1. / torch.tensor([piece_cnt[p] for p in PIECES], dtype=torch.float32)
    color_weights = 1. / torch.tensor([color_cnt[c] for c in COLORS], dtype=torch.float32)
    crit_piece = torch.nn.CrossEntropyLoss(weight=piece_weights.to(device))
    crit_color = torch.nn.CrossEntropyLoss(weight=color_weights.to(device))

    model = ChessCNN(len(PIECES), len(COLORS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_color = 0.
    best_piece = 0.
    best = 0.
    for epoch in range(30):
        model.train()
        running_loss, rp, rc, total = 0., 0., 0., 0
        for x, yp, yc in train_loader:
            x, yp, yc = x.to(device), yp.to(device), yc.to(device)
            optimizer.zero_grad()
            pp, pc = model(x)
            loss = crit_piece(pp, yp) + crit_color(pc, yc)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rp += (pp.argmax(1) == yp).sum().item()
            rc += (pc.argmax(1) == yc).sum().item()
            total += yp.size(0)
        scheduler.step()
        piece_acc, color_acc = rp/total, rc/total
        val_pa, val_ca = accuracy(model, val_loader, device)
        joint = (piece_acc + color_acc)/2
        print(f"epoch {epoch+1:02d}  loss {running_loss/len(train_loader):.3f}  "
              f"train C{color_acc:.3f} P{piece_acc:.3f}  "
              f"val C{val_ca:.3f} P{val_pa:.3f}")
        if joint > best:
            best = joint
            best_piece = piece_acc
            best_color = color_acc
            torch.save(model.state_dict(), "chess_best.pth")
    print("Training done – best weights saved to chess_best.pth", f"{best} = {best_color} {best_piece}")

if __name__ == "__main__":
    main()