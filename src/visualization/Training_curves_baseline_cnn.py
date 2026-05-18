import json
import matplotlib.pyplot as plt
from   pathlib import Path
import numpy as np
import seaborn as sns

# Load training history
experiments_dir = Path(__file__).parent.parent / "Experiments"
history_path = experiments_dir / "training_history.json"

# Check if training history exists
if not history_path.exists():
    print(f"⚠ Training history not found at: {history_path}")
    print("\nOptions:")
    print("1. Generate sample/template training data")
    print("2. Create training_history.json manually")
    print("\nCreating sample data for demonstration...\n")
    
    # Create sample training data
    num_epochs = 30
    # Simulated training loss (decreasing trend)
    epoch_losses = [0.5 - (0.45 * (i / num_epochs)) + np.random.normal(0, 0.02) for i in range(num_epochs)]
    epoch_losses = [max(0.05, loss) for loss in epoch_losses]  # Ensure positive
    
    # Simulated validation accuracy (increasing trend)
    val_accuracies = [55 + (35 * (i / num_epochs)) + np.random.normal(0, 1.5) for i in range(num_epochs)]
    val_accuracies = [min(100, max(50, acc)) for acc in val_accuracies]  # Keep between 50-100
    
    training_history = {
        "epoch_losses": epoch_losses,
        "val_accuracies": val_accuracies,
        "num_epochs": num_epochs
    }
    
    print("✓ Sample training data generated")
    print("  Note: This is sample data. To use your actual training metrics,")
    print(f"  re-run Train.py or manually create {history_path}\n")
else:
    # Load the history
    with open(history_path, 'r') as f:
        training_history = json.load(f)
    print(f"✓ Training history loaded from: {history_path}\n")

epoch_losses = training_history['epoch_losses']
val_accuracies = training_history['val_accuracies']
num_epochs = training_history['num_epochs']

# Create epochs array
epochs = np.arange(1, num_epochs + 1)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Define the light pastel blue color palette
colors = sns.light_palette("steelblue", 2)

# Plot training loss
ax1.plot(epochs, epoch_losses, color=colors[1], linewidth=2.5, marker='o', markersize=4, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)

# Add background color
ax1.set_facecolor(colors[0])
fig.patch.set_facecolor('white')

# Plot validation accuracy
ax2.plot(epochs, val_accuracies, color=colors[1], linewidth=2.5, marker='o', markersize=4, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)

# Add background color
ax2.set_facecolor(colors[0])

# Add max accuracy annotation
max_acc = max(val_accuracies)
max_epoch = val_accuracies.index(max_acc) + 1
ax2.annotate(f'Max: {max_acc:.2f}%\n(Epoch {max_epoch})', 
             xy=(max_epoch, max_acc), 
             xytext=(max_epoch + 2, max_acc - 2),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[1], alpha=0.7),
             arrowprops=dict(arrowstyle='->', color=colors[1], lw=1.5))

plt.tight_layout()

# Save the figure
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

output_path = results_dir / "training_curves.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Training curves saved to: {output_path}")

# Also create a combined view
fig2, ax = plt.subplots(figsize=(12, 6))

# Create secondary y-axis
ax2_twin = ax.twinx()

# Plot loss on primary axis
line1 = ax.plot(epochs, epoch_losses, color=colors[1], linewidth=2.5, marker='o', markersize=5, label='Training Loss')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color=colors[1])
ax.tick_params(axis='y', labelcolor=colors[1])

# Plot accuracy on secondary axis
line2 = ax2_twin.plot(epochs, val_accuracies, color='#A8D8EA', linewidth=2.5, marker='s', markersize=5, label='Validation Accuracy')
ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#A8D8EA')
ax2_twin.tick_params(axis='y', labelcolor='#A8D8EA')

ax.set_title('Training Loss and Validation Accuracy', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor(colors[0])
fig2.patch.set_facecolor('white')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', fontsize=11)

plt.tight_layout()

# Save the combined figure
output_path2 = results_dir / "training_curves_combined.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Combined training curves saved to: {output_path2}")

# Print summary statistics
print("\n" + "="*60)
print("Training Summary")
print("="*60)
print(f"Total Epochs: {num_epochs}")
print(f"Initial Loss: {epoch_losses[0]:.4f}")
print(f"Final Loss: {epoch_losses[-1]:.4f}")
print(f"Loss Reduction: {epoch_losses[0] - epoch_losses[-1]:.4f}")
print(f"Initial Accuracy: {val_accuracies[0]:.2f}%")
print(f"Final Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Best Accuracy: {max(val_accuracies):.2f}% (Epoch {val_accuracies.index(max(val_accuracies)) + 1})")
print("="*60)

plt.show()
