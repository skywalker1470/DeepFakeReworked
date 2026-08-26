import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from PIL import Image
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "./output/best_model.pth"
TEST_LIST = "./data_list/celebdf_test.txt"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Dataset
# ----------------------------
class TestDataset(Dataset):
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

        video_id = os.path.dirname(path)
        return img, label, video_id


# ----------------------------
# Transforms
# ----------------------------
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = TestDataset(TEST_LIST, test_tf)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Load Model
# ----------------------------
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Inference
# ----------------------------
video_probs = defaultdict(list)
video_labels = {}

with torch.no_grad():
    for imgs, labels, video_ids in loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        for prob, label, vid in zip(probs, labels, video_ids):
            video_probs[vid].append(prob)
            video_labels[vid] = label

# ----------------------------
# Video-level averaging
# ----------------------------
final_probs = []
final_labels = []

for vid in video_probs:
    avg_prob = np.mean(video_probs[vid])
    final_probs.append(avg_prob)
    final_labels.append(video_labels[vid])

final_probs = np.array(final_probs)
final_labels = np.array(final_labels)

# ----------------------------
# Default threshold (0.5)
# ----------------------------
preds_default = (final_probs > 0.5).astype(int)

acc = accuracy_score(final_labels, preds_default)
auc = roc_auc_score(final_labels, final_probs)
cm = confusion_matrix(final_labels, preds_default)

print("\n===== VIDEO LEVEL RESULTS =====")
print("Accuracy:", acc)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)

# ----------------------------
# Threshold tuning
# ----------------------------
best_acc = 0
best_thresh = 0

for t in np.arange(0.01, 0.99, 0.01):
    preds = (final_probs > t).astype(int)
    temp_acc = accuracy_score(final_labels, preds)
    if temp_acc > best_acc:
        best_acc = temp_acc
        best_thresh = t

print("\n===== THRESHOLD TUNING =====")
print("Best Threshold:", best_thresh)
print("Best Accuracy:", best_acc)

# Final confusion matrix
final_preds = (final_probs > best_thresh).astype(int)
final_cm = confusion_matrix(final_labels, final_preds)

print("\n===== FINAL CONFUSION MATRIX =====")
print(final_cm)