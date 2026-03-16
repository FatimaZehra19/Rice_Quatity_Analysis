import torch
import torch.nn as nn
import torch.optim as optim
from   Dataset_loader import train_loader, val_loader
from   Models import RiceCNN
from   tqdm import tqdm
from   pathlib import Path
from   datetime import datetime
import random
import numpy as np

# ========== SEED FOR REPRODUCIBILITY ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Seed set to: {SEED}")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = RiceCNN()
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
epoch_losses = []
val_accuracies = []
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
# Prepare experiments directory before training loop
experiments_dir = Path(__file__).parent.parent / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)

# Save best model
if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_path = experiments_dir / f"rice_cnn_baseline_best.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ New best model saved! (Accuracy: {val_accuracy:.2f}%)")

print("\n" + "="*60)
print("Training completed!")
print("="*60)

print("\nEpoch Losses:")
for i, loss in enumerate(epoch_losses):
    print(f"Epoch {i+1}: Loss = {loss:.4f}, Val Accuracy = {val_accuracies[i]:.2f}%")

print(f"\nBest Validation Accuracy: {max(val_accuracies):.2f}%")
print("="*60)

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = experiments_dir / f"rice_cnn_baseline_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"\nFinal model saved to: {model_path}")
print(f"Best model saved to: {experiments_dir / 'rice_cnn_baseline_best.pth'}")