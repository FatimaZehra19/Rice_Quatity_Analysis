"""
generate_training_curves.py
===========================
Generates training curves (loss and validation accuracy) for all 3 models.
Creates 3 PNG images from training history JSON files.

"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Set matplotlib style (with fallback if style not available)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

def generate_training_curves(model_name, history_file, output_file):
    """
    Generate training and validation curves for a model.

    Args:
        model_name (str): Display name of the model
        history_file (str): Path to training history JSON file (can use wildcards)
        output_file (str): Path to save PNG image
    """

    # Handle wildcard patterns
    if '*' in history_file:
        files = glob.glob(history_file)
        if not files:
            print(f"[ERROR] No history file found for pattern: {history_file}")
            return
        history_file = files[0]  # Use first match

    # Load training history
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] History file not found: {history_file}")
        return

    # Extract data
    losses = history.get('epoch_losses', [])
    val_accuracies = history.get('val_accuracies', [])

    if not losses or not val_accuracies:
        print(f"[ERROR] Invalid history data in {history_file}")
        return

    epochs = list(range(1, len(losses) + 1))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== PLOT 1: Training Loss =====
    ax1.plot(epochs, losses, 'r-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, len(epochs) + 1)

    # Add best loss annotation
    best_loss_idx = np.argmin(losses)
    ax1.annotate(f'Best: {losses[best_loss_idx]:.4f}',
                xy=(epochs[best_loss_idx], losses[best_loss_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=10)

    # ===== PLOT 2: Validation Accuracy =====
    ax2.plot(epochs, val_accuracies, 'b-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} - Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, len(epochs) + 1)
    ax2.set_ylim(min(val_accuracies) - 2, 100)

    # Add best accuracy annotation
    best_acc_idx = np.argmax(val_accuracies)
    ax2.annotate(f'Best: {val_accuracies[best_acc_idx]:.2f}%',
                xy=(epochs[best_acc_idx], val_accuracies[best_acc_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"[SUCCESS] Generated: {output_file}")
    print(f"  - Best loss: {losses[best_loss_idx]:.4f} (Epoch {epochs[best_loss_idx]})")
    print(f"  - Best accuracy: {val_accuracies[best_acc_idx]:.2f}% (Epoch {epochs[best_acc_idx]})")
    print()

def main():
    """Generate training curves for all 3 models."""

    # Define models and their history files
    models = [
        {
            "name": "Baseline CNN",
            "history": "Experiments/history/training_history.json",
            "output": "Results/training_curves/training_curves_baselineCNN.png"
        },
        {
            "name": "MobileNetV2",
            "history": "Experiments/history/training_history_mobilenetv2_*.json",
            "output": "Results/training_curves/training_curves_mobilenet.png"
        },
        {
            "name": "ResNet50",
            "history": "Experiments/history/training_history_resnet50_transfer_20260316_083810.json",
            "output": "Results/training_curves/training_curves_resnet.png"
        }
    ]

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    print("=" * 70)
    print("GENERATING TRAINING CURVES FOR ALL MODELS")
    print("=" * 70)
    print()

    # Generate curves for each model
    for model in models:
        history_path = str(project_root / model["history"])
        output_path = str(project_root / model["output"])

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {model['name']}")
        generate_training_curves(model["name"], history_path, output_path)

    print("=" * 70)
    print("ALL TRAINING CURVES GENERATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
