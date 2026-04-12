# dataset.py - X-ray dataset for TB detection
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# mild augmentation for training (medical-safe)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# clean transform for validation and inference
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


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

        self.transform = transform or val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)  # (3, 256, 256)
        return img, torch.tensor(label, dtype=torch.float32)
