# dataset.py - X-ray dataset for TB detection
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class XrayDataset(Dataset):
    def __init__(self, root_dir='../../data/preprocessed/xray',
                 classes=('tb_negative', 'tb_positive'),
                 transform=None):
        self.samples = []
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing directory: {cls_dir}")
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fn), float(label)))

        if not self.samples:
            raise RuntimeError(f"No image files found under {root_dir}")

        # if no custom transform, use default for EfficientNet-B0
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert('RGB')  # grayscale → 3 channel
        img = self.transform(img)              # (3, 224, 224)

        return img, torch.tensor(label, dtype=torch.float32)
