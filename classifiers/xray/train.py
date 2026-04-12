#!/bin/python3
# train.py - X-ray classifier training for TB detection
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from dataset import XrayDataset, train_transform, val_transform
from model import XrayClassifier


def main():
    # ===== CONFIGURATION =====
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print("-" * 50)

    # ===== DATASET =====
    print("Loading dataset...")
    train_ds = XrayDataset(root_dir='../../data/preprocessed/xray',
                           classes=('tb_negative', 'tb_positive'),
                           transform=train_transform)

    # Stratified 80/20 split
    negative_indices = [i for i in range(len(train_ds)) if train_ds[i][1] == 0]
    positive_indices = [i for i in range(len(train_ds)) if train_ds[i][1] == 1]

    print(f"Total: {len(train_ds)} samples")
    print(f"  Negative: {len(negative_indices)} ({len(negative_indices)/len(train_ds):.1%})")
    print(f"  Positive: {len(positive_indices)} ({len(positive_indices)/len(train_ds):.1%})")

    np.random.shuffle(negative_indices)
    np.random.shuffle(positive_indices)

    val_ratio = 0.20
    neg_val_count = int(val_ratio * len(negative_indices))
    pos_val_count = int(val_ratio * len(positive_indices))

    val_indices   = negative_indices[:neg_val_count] + positive_indices[:pos_val_count]
    train_indices = negative_indices[neg_val_count:] + positive_indices[pos_val_count:]

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_subset = Subset(train_ds, train_indices)

    # val subset gets clean transform, no augmentation
    val_ds = XrayDataset(root_dir='../../data/preprocessed/xray',
                         classes=('tb_negative', 'tb_positive'),
                         transform=val_transform)
    val_subset = Subset(val_ds, val_indices)

    print(f"\nTraining samples:   {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    neg_set = set(negative_indices)
    neg_train = sum(1 for i in train_indices if i in neg_set)
    pos_train = len(train_indices) - neg_train
    neg_val   = sum(1 for i in val_indices if i in neg_set)
    pos_val   = len(val_indices) - neg_val

    print(f"Train set:      Neg={neg_train} ({neg_train/len(train_indices):.1%}), "
          f"Pos={pos_train} ({pos_train/len(train_indices):.1%})")
    print(f"Validation set: Neg={neg_val} ({neg_val/len(val_indices):.1%}), "
          f"Pos={pos_val} ({pos_val/len(val_indices):.1%})")
    print("-" * 50)

    num_workers = int(os.environ.get('NUM_WORKERS', 2))
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False, num_workers=num_workers)

    # ===== MODEL =====
    model = XrayClassifier().to(device)

    # Initially freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    print("Backbone frozen initially — will be unfrozen later if configured")

    if os.path.exists('../../weights/xray_classifier.pt'):
        model.load_state_dict(torch.load('../../weights/xray_classifier.pt', map_location=device))
        print("Loaded previous best model (xray_classifier.pt)")
    else:
        print("Starting from scratch")

    # Weighted loss to handle class imbalance
    pos_weight = torch.tensor([neg_train / pos_train]).to(device)
    print(f"Class imbalance ratio: {neg_train / pos_train:.2f} — applying pos_weight")

    criterion = nn.BCEWithLogitsLoss() # Do initial run with pos_weight, then another run without it. forces model to learn positive class correctly first, then second run fine tunes the loss.

    # Initial optimizer (only classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=7)

    # ===== TRAINING PARAMS =====
    EPOCHS          = 35
    PATIENCE        = 35
    COLLAPSE_MARGIN = 0.10
    UNFREEZE_EPOCH  = 35          # set to > EPOCHS to never unfreeze; lower to enable

    best_val_acc      = 0.0
    epochs_no_improve = 0
    training_history  = []
    unfrozen          = False

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # ----- Unfreeze backbone if configured -----
        if not unfrozen and epoch >= UNFREEZE_EPOCH:
            for param in model.features.parameters():
                param.requires_grad = True
            # Recreate optimizer with lower LR for full fine-tuning
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                              factor=0.5, patience=7)
            print(f"\nEpoch {epoch+1}: Backbone unfrozen — fine-tuning entire model at lr=1e-4")
            unfrozen = True

        # Training phase
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            loss = criterion(model.classify(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase – compute both accuracy and loss (loss as extra metric)
        model.eval()
        val_correct = 0
        val_total   = 0
        val_losses  = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).float()

                logits = model.classify(x)
                loss = criterion(logits, y.unsqueeze(1))
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0

        # Collapse detection (still based on accuracy)
        drop_amount = best_val_acc - val_acc
        if drop_amount > COLLAPSE_MARGIN:
            print(f"\nModel Collapse! Epoch {epoch+1}")
            print(f"  Drop: {best_val_acc:.1%} → {val_acc:.1%} ({drop_amount:.1%} drop)")

            prev_checkpoint = f'checkpoint_epoch_{epoch}.pt'
            if epoch > 0 and os.path.exists(prev_checkpoint):
                model.load_state_dict(torch.load(prev_checkpoint, map_location=device))
                print(f"  Rolled back to epoch {epoch} model")
                val_acc = best_val_acc
            else:
                print("  No previous checkpoint to roll back to")

        # Checkpoint saving
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pt')

        # Best model saving (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../../weights/xray_classifier.pt')
            improvement_flag  = "New Best"
            epochs_no_improve = 0
        else:
            improvement_flag   = ""
            epochs_no_improve += 1

        # Learning rate scheduling (based on validation accuracy)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"Epoch {epoch+1:3d}: reducing learning rate to {new_lr:.2e}.")

        # Print both accuracy and validation loss
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.1%} {improvement_flag}")

        training_history.append({
            'epoch':      epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
            'val_acc':    val_acc,
            'best':       val_acc == best_val_acc
        })

        if epochs_no_improve >= PATIENCE:
            print(f"\nEARLY STOPPING at epoch {epoch+1}")
            print(f"  No improvement for {PATIENCE} epochs")
            print(f"  Best validation accuracy: {best_val_acc:.1%}")
            break

    # ===== WRAP UP =====
    print("\n" + "=" * 60)
    print("Training completed")
    print(f"Total epochs trained: {len(training_history)}")
    print(f"Best validation accuracy: {best_val_acc:.1%}")

    history_str = "Epoch,Train_Loss,Val_Loss,Val_Acc,Best\n"
    for entry in training_history:
        history_str += f"{entry['epoch']},{entry['train_loss']:.4f},{entry['val_loss']:.4f},{entry['val_acc']:.4f},{entry['best']}\n"

    with open('training_history.csv', 'w') as f:
        f.write(history_str)
    print("Training history saved as: training_history.csv")


if __name__ == '__main__':
    main()
