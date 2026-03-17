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

def load_history(path):
    if not path or not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)

def get_latest_history(pattern):
    files = glob.glob(str(experiments_dir / pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return Path(files[-1])

# 1. Load Histories
baseline_path = experiments_dir / "training_history.json"
resnet_path = get_latest_history("training_history_resnet50_transfer_*.json")
mobilenet_path = get_latest_history("training_history_mobilenetv2_transfer_*.json")

baseline_hist = load_history(baseline_path)
resnet_hist = load_history(resnet_path)
mobilenet_hist = load_history(mobilenet_path)

if not any([baseline_hist, resnet_hist, mobilenet_hist]):
    print("❌ Error: No training history found.")
    exit(1)

# ========== PLOTTING COMPARISON ==========
print("📊 Generating 3-Model Accuracy Comparison...")

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Function to plot if history exists
def plot_model(hist, label, color, marker, xytext_offset=(5, 5)):
    if hist:
        accs = hist['val_accuracies']
        epochs = np.arange(1, len(accs) + 1)
        plt.plot(epochs, accs, marker=marker, label=label, color=color, linewidth=2.5, markersize=6)
        # Annotate final accuracy with custom offset to prevent overlapping
        plt.annotate(f"{accs[-1]:.2f}%", (len(accs), accs[-1]), 
                     xytext=xytext_offset, textcoords='offset points', 
                     color=color, fontweight='bold', fontsize=10)
        return len(accs)
    return 0

max_epochs = 0
max_epochs = max(max_epochs, plot_model(baseline_hist, 'Baseline CNN', '#3498db', 'o', xytext_offset=(5, -15)))
max_epochs = max(max_epochs, plot_model(resnet_hist, 'ResNet50 Transfer', '#2ecc71', 's', xytext_offset=(5, 10)))
max_epochs = max(max_epochs, plot_model(mobilenet_hist, 'MobileNetV2 Transfer', '#f39c12', '^', xytext_offset=(5, -5)))

plt.title('Validation Accuracy Comparison: All Models', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.xlim(0.5, max_epochs + 1.5) # Leave space for annotations

plt.tight_layout()
output_path = results_dir / "all_models_accuracy_comparison.png"
plt.savefig(output_path, dpi=300)
print(f"✅ Success! Combined comparison graph saved to: {output_path}")

plt.show()
