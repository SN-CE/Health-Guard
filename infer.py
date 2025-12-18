#!/bin/python3
# infer.py — compact inference with accuracy + ROC-AUC

import os
import torch
import numpy as np
from model import SmallCNN
from sklearn.metrics import roc_auc_score

# -------------------------------
# 1. Device setup
# -------------------------------
torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------
# 2. Load trained model
# -------------------------------
model = SmallCNN().to(device)
if os.path.exists("tb_guardian.pt"):
    model.load_state_dict(torch.load("tb_guardian.pt", map_location=device))
    model.eval()
    print("Loaded trained weights (tb_guardian.pt)")
else:
    raise FileNotFoundError("No trained model found (tb_guardian.pt)")

# -------------------------------
# 3. Collect preprocessed features & labels
# -------------------------------
test_root = "preprocessed_test_data"

samples = []
for label_name, label_val in [("tb_negative", 0), ("tb_positive", 1)]:
    dir_path = os.path.join(test_root, label_name)
    if not os.path.isdir(dir_path):
        continue
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".npy"):
                samples.append({
                    "path": os.path.join(root, f),
                    "label": label_val,
                    "name": f
                })

if not samples:
    raise FileNotFoundError("No .npy files found in preprocessed_test_data/")

# -------------------------------
# 4. Inference loop with accuracy tracking
# -------------------------------
correct = 0
total = 0
all_probs = []
all_labels = []

print(f"\n{'FILE':35s} | ACTUAL | PRED | PROB")
print("-" * 60)

for s in samples:
    feat = np.load(s["path"])
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item() * 100  # convert to %

    all_probs.append(prob / 100.0)
    all_labels.append(s["label"])

    pred_label = 1 if prob >= 50 else 0
    actual_label = s["label"]

    if pred_label == actual_label:
        correct += 1
    total += 1

    actual_str = "TB+" if actual_label == 1 else "TB-"
    pred_str = "TB+" if pred_label == 1 else "TB-"

    print(f"{s['name'][:35]:35s} | {actual_str:^6s} | {pred_str:^4s} | {prob:5.1f}")

# -------------------------------
# 5. Overall metrics
# -------------------------------
acc = (correct / total) * 100 if total > 0 else 0
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = 0.5  # fallback if only one class in test set

print("-" * 60)
print(f"Total files: {total} | Accuracy: {acc:.2f}% | ROC-AUC: {auc:.4f}")
