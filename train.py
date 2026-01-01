#!/bin/python3
# train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score
from dataset import CoughDataset
from model import SmallCNN

torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name()}")

ds = CoughDataset(root_dir='preprocessed_train_data', classes=('tb_negative','tb_positive'))
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)

model = SmallCNN().to(device)

if os.path.exists("tb_guardian.pt"):
	model.load_state_dict(torch.load("tb_guardian.pt", map_location=device))
	print("Weights Loaded")
else:
	print("Training From Scratch")

criterion = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for ep in range(epochs):
	model.train()
	losses = []
	for x,y in loader:
		x = x.to(device).float()
		y = y.to(device).float()
		opt.zero_grad()
		logits = model(x)
		loss = criterion(logits, y)
		loss.backward()
		opt.step()
		losses.append(loss.item())

	# quick train-set AUC (fast check)
	model.eval()
	ys, ps = [], []
	with torch.no_grad():
		for x,y in loader:
			x = x.to(device).float()
			logits = model(x)
			probs = torch.sigmoid(logits).cpu().numpy()
			ys.extend(y.numpy().tolist())
			ps.extend(probs.tolist())
	auc = roc_auc_score(ys, ps) if len(set(ys))>1 else 0.5
	print(f'Epoch {ep+1}/{epochs} loss={sum(losses)/len(losses):.4f} auc={auc:.4f}')

torch.save(model.state_dict(), 'tb_guardian.pt')

