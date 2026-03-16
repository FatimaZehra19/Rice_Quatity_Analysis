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

# ========== PLOTTING COMPARISON ==========
if baseline_history and resnet_history:
    print("\n📊 Generating Model Comparison Graph...")
    
    b_accs = baseline_history['val_accuracies']
    r_accs = resnet_history['val_accuracies']
    
    # Take the minimum length in case they differ
    min_epochs = min(len(b_accs), len(r_accs))
    epochs = np.arange(1, min_epochs + 1)
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(epochs, b_accs[:min_epochs], 'o-', label='Baseline CNN', linewidth=2.5, markersize=6)
    plt.plot(epochs, r_accs[:min_epochs], 's-', label='ResNet50 Transfer', linewidth=2.5, markersize=6)
    
    plt.title('Accuracy Comparison: Baseline vs ResNet50', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Highlight final accuracies
    plt.annotate(f"{b_accs[-1]:.2f}%", (min_epochs, b_accs[-1]), xytext=(min_epochs-2, b_accs[-1]-2), 
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=5))
    plt.annotate(f"{r_accs[-1]:.2f}%", (min_epochs, r_accs[-1]), xytext=(min_epochs-2, r_accs[-1]+2), 
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison_accuracy.png", dpi=300)
    print(f"✓ Saved Comparison graph to: Results/model_comparison_accuracy.png")

print("\n✅ Visualization complete!")
