"""
Train_All_Models.py
===================
Trains all three models sequentially and saves weights to Experiments/.

    Baseline CNN      -> Experiments/rice_cnn_baseline_best.pth
    MobileNetV2       -> Experiments/rice_mobilenetv2_transfer_best.pth
    ResNet50          -> Experiments/rice_resnet50_transfer_best.pth

HOW TO RUN
----------
Local (after setting DATASET_PATH below):
    python Train_All_Models.py

Google Colab:
    1. Upload your project zip to Drive, then mount it:
         from google.colab import drive
         drive.mount('/content/drive')
    2. Unzip:
         !unzip /content/drive/MyDrive/Rice_thesis_project.zip -d /content/
    3. Install deps:
         !pip install torch torchvision tqdm
    4. Run:
         !python /content/Rice_thesis_project/Train_All_Models.py

Kaggle:
    1. Upload dataset via Kaggle Datasets tab (or use the Kaggle rice dataset directly)
    2. Set DATASET_PATH = "/kaggle/input/rice-image-dataset/Rice_Image_Dataset"
    3. Run as a Kaggle notebook cell: exec(open("Train_All_Models.py").read())

GPU tip: if you have a GPU, increase BATCH_SIZE to 64 or 128 for faster training.
"""

# =============================================================================
# CHANGE THIS PATH TO MATCH YOUR MACHINE
# =============================================================================
DATASET_PATH = r"D:\Projects\Rice_thesis_project\Dataset\Rice_Image_Dataset"
# =============================================================================

import os
import sys
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ── output directory ──────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
EXPERIMENTS    = PROJECT_ROOT / "Experiments"
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── training hyperparameters ─────────────────────────────────────────────────
NUM_EPOCHS    = 30
BATCH_SIZE    = 32    # increase to 64 or 128 on a GPU
LR            = 0.001
WEIGHT_DECAY  = 1e-4
LR_STEP       = 10    # StepLR: drop LR every N epochs
LR_GAMMA      = 0.5   # StepLR: multiply LR by this factor
NUM_CLASSES   = 5
HIDDEN_UNITS  = 512
DROPOUT       = 0.5


# =============================================================================
# 1. DATASET
# =============================================================================

def build_dataloaders(dataset_path, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    n = len(full_dataset)
    train_n = int(0.70 * n)
    val_n   = int(0.15 * n)
    test_n  = n - train_n - val_n

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_n, val_n, test_n], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Dataset : {n:,} images  |  Classes: {full_dataset.classes}")
    print(f"  Split   : {train_n:,} train  /  {val_n:,} val  /  {test_n:,} test")

    return train_loader, val_loader, test_loader, full_dataset.classes


# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

class RiceCNN(nn.Module):
    """Custom 4-layer CNN baseline."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32);  self.drop1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1); self.bn2 = nn.BatchNorm2d(64);  self.drop2 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128); self.drop3 = nn.Dropout2d(0.25)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256); self.drop4 = nn.Dropout2d(0.25)
        self.pool         = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1     = nn.Linear(256, HIDDEN_UNITS)
        self.fc_drop = nn.Dropout(DROPOUT)
        self.fc2     = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = self.pool(self.drop1(self.bn1(F.relu(self.conv1(x)))))
        x = self.pool(self.drop2(self.bn2(F.relu(self.conv2(x)))))
        x = self.pool(self.drop3(self.bn3(F.relu(self.conv3(x)))))
        x = self.pool(self.drop4(self.bn4(F.relu(self.conv4(x)))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_mobilenetv2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.last_channel, HIDDEN_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_UNITS, NUM_CLASSES),
    )
    return model, model.classifier.parameters()


def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, HIDDEN_UNITS),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_UNITS, NUM_CLASSES),
    )
    return model, model.fc.parameters()


# =============================================================================
# 3. TRAINING ENGINE
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def train_model(name, model, trainable_params, train_loader, val_loader, save_name):
    """Full training loop with checkpointing. Returns training history dict."""

    print(f"\n{'='*65}")
    print(f"  Training: {name}")
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_p:,} total  |  {trainable_p:,} trainable")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    history = {
        "model": name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seed": SEED,
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS, "learning_rate": LR,
            "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
            "lr_step": LR_STEP, "lr_gamma": LR_GAMMA,
            "dropout": DROPOUT, "hidden_units": HIDDEN_UNITS,
        },
        "train_losses": [], "train_accuracies": [],
        "val_losses":   [], "val_accuracies":   [],
        "best_val_accuracy": 0.0, "best_epoch": 0,
        "total_params": total_p, "trainable_params": trainable_p,
    }

    best_val_acc = 0.0
    best_path    = EXPERIMENTS / save_name
    start_time   = time.time()

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step()

        history["train_losses"].append(round(train_loss, 6))
        history["train_accuracies"].append(round(train_acc, 4))
        history["val_losses"].append(round(val_loss, 6))
        history["val_accuracies"].append(round(val_acc, 4))

        # Save best model (inside the loop — fixed checkpointing bug)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc          = val_acc
            history["best_val_accuracy"] = round(best_val_acc, 4)
            history["best_epoch"]        = epoch + 1
            torch.save(model.state_dict(), best_path)
            marker = "  <-- best saved"

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}{marker}"
        )

    # Save training history JSON
    ts           = history["timestamp"]
    history_name = f"training_history_{save_name.replace('.pth', '')}_{ts}.json"
    history_path = EXPERIMENTS / history_name
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n  Done in {elapsed_min:.1f} min")
    print(f"  Best val accuracy : {best_val_acc:.2f}%  (epoch {history['best_epoch']})")
    print(f"  Model saved       : {best_path}")
    print(f"  History saved     : {history_path}")

    return history


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  RICE CLASSIFICATION — FULL TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Dataset : {DATASET_PATH}")
    print(f"  Output  : {EXPERIMENTS}")
    print(f"  Epochs  : {NUM_EPOCHS}   |   Batch: {BATCH_SIZE}   |   Seed: {SEED}")
    print("=" * 65)

    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at:\n  {DATASET_PATH}")
        print("Update the DATASET_PATH variable at the top of this file.")
        sys.exit(1)

    print("\n[Loading dataset]")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(DATASET_PATH)

    all_histories = {}

    # ── Model 1: Baseline CNN ─────────────────────────────────────────────────
    baseline = RiceCNN()
    h1 = train_model(
        name            = "Baseline CNN",
        model           = baseline,
        trainable_params= baseline.parameters(),   # all params trained from scratch
        train_loader    = train_loader,
        val_loader      = val_loader,
        save_name       = "rice_cnn_baseline_best.pth",
    )
    all_histories["baseline_cnn"] = h1

    # ── Model 2: MobileNetV2 Transfer Learning ────────────────────────────────
    mobilenet, mobilenet_params = build_mobilenetv2()
    h2 = train_model(
        name            = "MobileNetV2 (Transfer Learning)",
        model           = mobilenet,
        trainable_params= mobilenet_params,
        train_loader    = train_loader,
        val_loader      = val_loader,
        save_name       = "rice_mobilenetv2_transfer_best.pth",
    )
    all_histories["mobilenetv2"] = h2

    # ── Model 3: ResNet50 Transfer Learning ───────────────────────────────────
    resnet, resnet_params = build_resnet50()
    h3 = train_model(
        name            = "ResNet50 (Transfer Learning)",
        model           = resnet,
        trainable_params= resnet_params,
        train_loader    = train_loader,
        val_loader      = val_loader,
        save_name       = "rice_resnet50_transfer_best.pth",
    )
    all_histories["resnet50"] = h3

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ALL MODELS TRAINED — SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<35} {'Best Val Acc':>12}  {'Best Epoch':>10}")
    print(f"  {'-'*60}")
    for key, h in all_histories.items():
        print(f"  {h['model']:<35} {h['best_val_accuracy']:>11.2f}%  {h['best_epoch']:>10}")

    print(f"\n  All weights saved to: {EXPERIMENTS}")
    print("=" * 65)
    print("\nNext step: copy the Experiments/ folder back to your project,")
    print("then run the evaluation scripts or launch Rice_App.py.")


if __name__ == "__main__":
    main()
