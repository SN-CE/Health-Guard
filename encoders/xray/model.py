# model.py - X-ray encoder for TB detection
import torch.nn as nn
from torchvision.models import efficientnet_b0


class XrayEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(pretrained=True)

        # strip the classifier, keep only the feature extractor
        self.features = base.features
        self.pool = base.avgpool

        # our own classifier head for standalone training
        self.classifier = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def encode(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (batch, 1280)

    def classify(self, x):
        return self.classifier(self.encode(x))  # (batch, 1)

    def forward(self, x):
        return self.encode(x)
