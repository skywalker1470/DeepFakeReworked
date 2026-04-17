import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Dataset
# ----------------------------
class CelebDFDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        self.transform = transform

        with open(txt_path, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default="./data_list/celebdf_train.txt")
    parser.add_argument("--val_list", default="./data_list/celebdf_val.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)  # smaller batch
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------
    # Data Augmentation
    # ----------------------------
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),  # smaller size
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CelebDFDataset(args.train_list, train_tf)
    val_dataset = CelebDFDataset(args.val_list, val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ----------------------------
    # EfficientNet-B0 (Lightweight)
    # ----------------------------
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    # Freeze first half
    total_layers = len(list(model.features))
    freeze_until = total_layers // 2

    for i, layer in enumerate(model.features):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False

    model = model.to(device)

    # ----------------------------
    # Weighted BCE Loss
    # ----------------------------
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler()  # AMP

    best_val_acc = 0.0

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = model(imgs)
                preds = torch.sigmoid(outputs) > 0.5

                correct += (preds.float() == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print("Model saved.")

    print("Training complete.")
    print("Best Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    main()