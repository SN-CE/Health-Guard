#!/bin/python3
# train.py - Safe training with auto-checkpoints and early stopping
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from dataset import CoughDataset
from model import SmallCNN

torch.backends.cudnn.enabled = False

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
print("-" * 50)

# Load training data
print("Loading dataset...")
train_ds = CoughDataset(root_dir='preprocessed_train_data',
                       classes=('tb_negative', 'tb_positive'))

# ===== STRATIFIED 80/20 SPLIT =====
print("Creating stratified 80/20 split...")

# Get indices for each class
negative_indices = []
positive_indices = []

for idx in range(len(train_ds)):
    _, label = train_ds[idx]
    if label == 0:
        negative_indices.append(idx)
    else:
        positive_indices.append(idx)

print(f"Total: {len(train_ds)} samples")
print(f"  Negative: {len(negative_indices)} ({len(negative_indices)/len(train_ds):.1%})")
print(f"  Positive: {len(positive_indices)} ({len(positive_indices)/len(train_ds):.1%})")

# Shuffle each class list
np.random.shuffle(negative_indices)
np.random.shuffle(positive_indices)

# Calculate split sizes (20% validation)
val_ratio = 0.20
neg_val_count = int(val_ratio * len(negative_indices))
pos_val_count = int(val_ratio * len(positive_indices))

# Create validation indices (first X from each shuffled list)
val_indices = (negative_indices[:neg_val_count] + 
               positive_indices[:pos_val_count])

# Create training indices (remaining files)
train_indices = (negative_indices[neg_val_count:] + 
                 positive_indices[pos_val_count:])

# Shuffle the final sets
np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

# Create subsets
train_subset = Subset(train_ds, train_indices)
val_subset = Subset(train_ds, val_indices)

print(f"\nTraining samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")

# Verify class distribution
def check_distribution(subset, name):
    neg_count = sum(1 for i in range(len(subset)) if subset[i][1] == 0)
    pos_count = len(subset) - neg_count
    print(f"{name}: Neg={neg_count} ({neg_count/len(subset):.1%}), "
          f"Pos={pos_count} ({pos_count/len(subset):.1%})")

check_distribution(train_subset, "Train set")
check_distribution(val_subset, "Validation set")
print("-" * 50)

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)

# Model setup
model = SmallCNN().to(device)

# Load existing model if available
if os.path.exists("tb_guardian.pt"):
    model.load_state_dict(torch.load("tb_guardian.pt", map_location=device))
    print("Loaded previous best model (tb_guardian.pt)")
else:
    print("Starting from scratch")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training parameters
EPOCHS = 250  # Change this number as needed
PATIENCE = 15  # Stop if no improvement for 5 epochs
COLLAPSE_MARGIN = 0.10  # If val acc drops 10% below best

# Trackers
best_val_acc = 0.0
epochs_no_improve = 0
training_history = []

# Clean up old checkpoints (except the best model)
print("Cleaning up old checkpoints...")
for f in os.listdir('.'):
    if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
        os.remove(f)

print("\nStarting training...")
print("=" * 60)

# Training loop
for epoch in range(EPOCHS):
    # ===== TRAINING PHASE =====
    model.train()
    train_losses = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device).float()
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # ===== VALIDATION PHASE =====
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x)
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()

            val_correct += (predictions == y).sum().item()
            val_total += len(y)

    val_acc = val_correct / val_total if val_total > 0 else 0

    # ===== COLLAPSE DETECTION =====
    drop_amount = best_val_acc - val_acc
    if drop_amount > COLLAPSE_MARGIN:
        print(f"\n⚠  COLLAPSE DETECTED! Epoch {epoch+1}")
        print(f"  Drop: {best_val_acc:.1%} → {val_acc:.1%} ({drop_amount:.1%} drop)")

        # Roll back to best model
        if epoch > 0 and os.path.exists(f'checkpoint_epoch_{epoch}.pt'):
            model.load_state_dict(torch.load(f'checkpoint_epoch_{epoch}.pt', map_location=device))
            print(f"   Rolled back to epoch {epoch} model")
            val_acc = best_val_acc  # Use previous best accuracy
        else:
            print("   No previous checkpoint to roll back to")

    # ===== CHECKPOINT SAVING =====
    # Save checkpoint for this epoch
    checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
    torch.save(model.state_dict(), checkpoint_path)

    # Save as best model if improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'tb_guardian.pt')
        torch.save(model.state_dict(), 'best_checkpoint.pt')
        improvement_flag = "✓ NEW BEST"
        epochs_no_improve = 0
    else:
        improvement_flag = ""
        epochs_no_improve += 1

    # ===== PROGRESS DISPLAY =====
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Acc: {val_acc:.1%} {improvement_flag}")

    # Save training history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_acc': val_acc,
        'best': val_acc == best_val_acc
    })

    # ===== EARLY STOPPING =====
    if epochs_no_improve >= PATIENCE:
        print(f"\n⏹  EARLY STOPPING at epoch {epoch+1}")
        print(f"   No improvement for {PATIENCE} epochs")
        print(f"   Best validation accuracy: {best_val_acc:.1%}")
        break

# ===== FINAL CLEANUP =====
print("\n" + "=" * 60)
print("Training completed!")

# Load the absolute best model
if os.path.exists('best_checkpoint.pt'):
    model.load_state_dict(torch.load('best_checkpoint.pt', map_location=device))
    print("Loaded best model from training")

# Final save
torch.save(model.state_dict(), 'tb_guardian.pt')
print(f"Best model saved as: tb_guardian.pt")

# Save training history
history_str = "Epoch,Train_Loss,Val_Acc,Best\n"
for entry in training_history:
    history_str += f"{entry['epoch']},{entry['train_loss']:.4f},{entry['val_acc']:.4f},{entry['best']}\n"

with open('training_history.csv', 'w') as f:
    f.write(history_str)
print("Training history saved as: training_history.csv")

# Quick summary
print("\n📊 SUMMARY:")
print(f"Total epochs trained: {len(training_history)}")
print(f"Best validation accuracy: {best_val_acc:.1%}")
print(f"Checkpoints saved: {len([f for f in os.listdir('.') if f.startswith('checkpoint_epoch_')])}")

# Clean up intermediate best checkpoint
if os.path.exists('best_checkpoint.pt'):
    os.remove('best_checkpoint.pt')

print("\n✅ Ready for inference! Run: python infer.py")
