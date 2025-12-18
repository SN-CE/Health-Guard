import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CoughDataset(Dataset):
    def __init__(self, root_dir='preprocessed_data',
                 classes=('tb_negative', 'tb_positive'),
                 transform=None):
        """
        Args:
            root_dir (str): Path containing subfolders for each class (tb_positive / tb_negative)
            classes (tuple): Folder names for each label
            transform (callable, optional): Torch transform applied to each feature
        """
        self.samples = []
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing directory: {cls_dir}")
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fn), float(label)))

        if not self.samples:
            raise RuntimeError(f"No .npy files found under {root_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load preprocessed feature (already numpy)
        feat = np.load(path).astype(np.float32)

        if self.transform:
            feat = self.transform(feat)

        # Add channel dimension (for CNN input)
        feat = torch.from_numpy(feat).unsqueeze(0)  # shape: (1, H, W)

        return feat, torch.tensor(label, dtype=torch.float32)
