import os, cv2, torch, collections, albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from ChessCNN import ChessCNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

DATASET_DIR = "../assets/chess_pieces"
PIECES = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]

# ---------- transforms ----------
train_tf = A.Compose([
    A.Resize(30, 30),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=10,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.4
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
    ToTensorV2()
])
#0.98


# ---------- dataset ----------
class ChessDataset(Dataset):
    def __init__(self, root, pieces, transform=None):
        self.samples = []
        self.p2i = {p: i for i, p in enumerate(pieces)}
        self.transform = transform

        for folder in os.listdir(root):
            path = os.path.join(root, folder)
            piece = folder.title()

            for fname in os.listdir(path):
                if fname.lower().endswith('.png'):
                    self.samples.append((os.path.join(path, fname), piece))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, piece = self.samples[idx]

        img = cv2.imread(path)[:, :, ::-1]

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1)

        return img, self.p2i[piece]


# ---------- helpers ----------
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

@staticmethod
def showChart(allEpoches):
    # Rozpakowujemy listę krotek na osobne listy
    # allEpoches zawiera [(loss1, t_acc1, v_acc1), (loss2, t_acc2, v_acc2), ...]
    losses, train_accs, val_accs = zip(*allEpoches)

    plt.figure(figsize=(10, 6))

    # Rysujemy każdą linię z etykietą (label), która trafi do legendy
    plt.plot(losses, label="Loss (Strata)", color="gray", linestyle="--")
    plt.plot(train_accs, label="Train Accuracy", color="blue", marker="o", markersize=4)
    plt.plot(val_accs, label="Val Accuracy", color="red", marker="x", markersize=4)

    plt.xlabel("Epoki")
    plt.ylabel("Wartość")
    plt.title("Proces uczenia modelu ChessCNN")

    # Aktywacja legendy
    plt.legend(loc="upper right")

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Ustawienia osi Y dla lepszej czytelności
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Zapis i pokazanie
    plt.savefig("training_chart.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(model, val_loader, device, pieces):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Obliczanie macierzy
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=pieces, columns=pieces)

    # Rysowanie
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Chess Pieces')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# ---------- main ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = ChessDataset(DATASET_DIR, PIECES, transform=train_tf)
    train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), len(ds) - int(0.8*len(ds))])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds , batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # piece class weights
    raw_indices = train_ds.indices
    piece_cnt = collections.Counter([ds.samples[i][1] for i in raw_indices])
    piece_weights = 1. / torch.tensor([piece_cnt[p] for p in PIECES], dtype=torch.float32)

    criterion = torch.nn.CrossEntropyLoss(weight=piece_weights.to(device))

    model = ChessCNN(num_pieces=len(PIECES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best = 0
    allEpoches = []
    for epoch in range(12):
        model.train()
        running_loss, correct, total = 0., 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_acc = correct / total
        val_acc = accuracy(model, val_loader, device)

        print(f"epoch {epoch+1:02d}  loss {running_loss/len(train_loader):.3f}  train {train_acc:.3f}  val {val_acc:.3f}")

        allEpoches.append((running_loss/len(train_loader), train_acc, val_acc))
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "chess_best.pth")

    print("Done – best model saved. Best val acc:", best)
    showChart(allEpoches)
    plot_confusion_matrix(model, val_loader, device, PIECES)


if __name__ == "__main__":
    main()