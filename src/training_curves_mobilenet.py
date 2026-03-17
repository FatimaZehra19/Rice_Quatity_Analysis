import json
import matplotlib.pyplot as plt
from pathlib import Path
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

# 2. Find Latest MobileNet History
mobilenet_pattern = str(experiments_dir / "training_history_mobilenetv2_transfer_*.json")
mobilenet_files = glob.glob(mobilenet_pattern)
mobilenet_files.sort(key=os.path.getmtime)
mobilenet_path = Path(mobilenet_files[-1]) if mobilenet_files else None

# ========== DATA LOADING ==========
def load_history(path):
    if not path or not path.exists():
        print(f"⚠ Warning: History not found at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

baseline_history = load_history(baseline_path)
mobilenet_history = load_history(mobilenet_path)

if not baseline_history and not mobilenet_history:
    print("❌ Error: No training history found. Please train models first.")
    exit(1)

# ========== PLOTTING MOBILENET CURVES ==========
if mobilenet_history:
    print(f"📈 Plotting MobileNetV2 Training Curves from: {mobilenet_path.name}")
    
    losses = mobilenet_history['epoch_losses']
    accs = mobilenet_history['val_accuracies']
    epochs = np.arange(1, len(losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors
    palette = sns.color_palette("magma", 2)
    
    # Loss Plot
    ax1.plot(epochs, losses, color=palette[0], linewidth=2.5, marker='o', label='MobileNetV2 Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('MobileNetV2: Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy Plot
    ax2.plot(epochs, accs, color=palette[1], linewidth=2.5, marker='s', label='MobileNetV2 Accuracy')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('MobileNetV2: Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "mobilenetv2_training_curves.png", dpi=300)
    print(f"✓ Saved MobileNet curves to: Results/mobilenetv2_training_curves.png")


print("\n✅ Visualization complete!")
