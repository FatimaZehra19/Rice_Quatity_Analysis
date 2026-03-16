import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Dataset_loader import train_loader, val_loader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import random

# ========== SEED FOR REPRODUCIBILITY ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ========== HYPERPARAMETERS ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 5
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
LR_SCHEDULER_STEP = 10
LR_SCHEDULER_GAMMA = 0.5
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 512

print("="*70)
print("RESNET50 TRANSFER LEARNING - RICE CLASSIFICATION")
print("="*70)
print("\n📊 HYPERPARAMETERS:")
print(f"  • Seed: {SEED}")
print(f"  • Device: {DEVICE}")
print(f"  • Number of Classes: {NUM_CLASSES}")
print(f"  • Number of Epochs: {NUM_EPOCHS}")
print(f"  • Learning Rate: {LEARNING_RATE}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Weight Decay: {WEIGHT_DECAY}")
print(f"  • Dropout Rate: {DROPOUT_RATE}")
print(f"  • Hidden Units: {HIDDEN_UNITS}")
print(f"  • LR Scheduler: StepLR (step={LR_SCHEDULER_STEP}, gamma={LR_SCHEDULER_GAMMA})")
print("="*70 + "\n")

# ========== TRANSFER LEARNING SETUP ==========
print("🔄 Loading pretrained ResNet50 (ImageNet weights)...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("✓ Loaded pretrained ResNet50")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"  Frozen parameters: {frozen_params:,}")

# Replace the final fully connected layer for rice classes
print(f"\n🎯 Replacing final layer with custom classifier for {NUM_CLASSES} rice classes...")
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, HIDDEN_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(HIDDEN_UNITS, NUM_CLASSES)
)

# Unfreeze the final layer for training
for param in model.fc.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {trainable_params:,}")

model = model.to(DEVICE)
print("✓ Model ready for training")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)


# ========== TRAINING SETUP ==========
epoch_losses = []
val_accuracies = []
best_val_acc = 0.0
best_epoch = 0

print("\n" + "="*70)
print("🚀 STARTING TRANSFER LEARNING TRAINING")
print("="*70 + "\n")

# ========== MAIN TRAINING LOOP ==========
for epoch in range(NUM_EPOCHS):
    print(f"\n📍 Epoch [{epoch+1}/{NUM_EPOCHS}]", end=" | ")
    
    # Training phase
    model.train()
    running_loss = 0.0
    train_samples = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        train_samples += labels.size(0)

    epoch_loss = running_loss / train_samples
    epoch_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_accuracies.append(val_accuracy)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Loss: {epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}")
    
    # Learning rate scheduling
    scheduler.step()
    
    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_epoch = epoch + 1
        experiments_dir = Path(__file__).parent.parent / "Experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = experiments_dir / "rice_resnet50_transfer_best.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"  ⭐ New best model saved! Accuracy: {val_accuracy:.2f}%")

# ========== SAVE TRAINING HISTORY ==========
experiments_dir = Path(__file__).parent.parent / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history = {
    "model": "ResNet50-Transfer-Learning",
    "timestamp": timestamp,
    "seed": SEED,
    "hyperparameters": {
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "hidden_units": HIDDEN_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_step": LR_SCHEDULER_STEP,
        "lr_scheduler_gamma": LR_SCHEDULER_GAMMA
    },
    "epoch_losses": epoch_losses,
    "val_accuracies": val_accuracies,
    "best_val_accuracy": float(best_val_acc),
    "best_epoch": best_epoch,
    "total_trainable_params": trainable_params,
    "total_params": total_params
}

history_path = experiments_dir / f"training_history_resnet50_transfer_{timestamp}.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)

print(f"\n✓ Training history saved to: {history_path}")

# ========== FINAL RESULTS SUMMARY ==========
print("\n" + "="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)

print("\n📊 RESULTS:")
print("-" * 70)
print(f"  Best Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"  Final Accuracy: {val_accuracies[-1]:.2f}%")
print(f"  Initial Loss: {epoch_losses[0]:.4f}")
print(f"  Final Loss: {epoch_losses[-1]:.4f}")
print(f"  Loss Reduction: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%")
print(f"  Avg Accuracy: {np.mean(val_accuracies):.2f}%")

print("\n💾 SAVED:")
print("-" * 70)
print(f"  • Model: rice_resnet50_transfer_best.pth")
print(f"  • History: training_history_resnet50_transfer_{timestamp}.json")
print("="*70)