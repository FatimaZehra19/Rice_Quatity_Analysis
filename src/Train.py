import torch
import torch.nn as nn
import torch.optim as optim
from Dataset_loader import train_loader, val_loader
from Models import RiceCNN
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

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
num_epochs = 1
epoch_losses = []
for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch+1}")
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training completed.")

print("\nEpoch Losses:")
for i, loss in enumerate(epoch_losses):
    print(f"Epoch {i+1}: {loss:.4f}")

# Save the trained model
experiments_dir = Path(__file__).parent.parent / "Experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = experiments_dir / f"rice_cnn_baseline_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully to: {model_path}")