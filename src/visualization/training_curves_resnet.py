import json
import matplotlib.pyplot as plt
from   pathlib import Path
import numpy as np
import seaborn as sns
import glob
import os

# ========== CONFIGURATION ==========
experiments_dir = Path(__file__).parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

# 1. Find Baseline History
baseline_path = experiments_dir / "training_history.json"

# 2. Find Latest ResNet History
resnet_pattern = str(experiments_dir / "training_history_resnet50_transfer_*.json")
resnet_files = glob.glob(resnet_pattern)
resnet_files.sort(key=os.path.getmtime)
resnet_path = Path(resnet_files[-1]) if resnet_files else None

# ========== DATA LOADING ==========
def load_history(path):
    if not path or not path.exists():
        print(f"⚠ Warning: History not found at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

baseline_history = load_history(baseline_path)
resnet_history = load_history(resnet_path)

if not baseline_history and not resnet_history:
    print("❌ Error: No training history found. Please train models first.")
    exit(1)

# ========== PLOTTING RESNET CURVES ==========
if resnet_history:
    print(f"📈 Plotting ResNet50 Training Curves from: {resnet_path.name}")
    
    losses = resnet_history['epoch_losses']
    accs = resnet_history['val_accuracies']
    epochs = np.arange(1, len(losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors
    palette = sns.color_palette("viridis", 2)
    
    # Loss Plot
    ax1.plot(epochs, losses, color=palette[0], linewidth=2.5, marker='o', label='ResNet50 Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('ResNet50: Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy Plot
    ax2.plot(epochs, accs, color=palette[1], linewidth=2.5, marker='s', label='ResNet50 Accuracy')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('ResNet50: Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "resnet50_training_curves.png", dpi=300)
    print(f"✓ Saved ResNet curves to: Results/resnet50_training_curves.png")


print("\n✅ Visualization complete!")
