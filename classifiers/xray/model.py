# model.py - X-ray classifier for TB detection
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class XrayClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

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

    def classify(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # (batch, 1)

    def forward(self, x):
        return self.classify(x)
