import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.Dataset_loader import get_data_loaders
from src.models.Baseline_CNN_Model import RiceCNN

# ========== SETUP ==========
_, _, test_loader, class_names = get_data_loaders(num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")

model = RiceCNN().to(device)
experiments_dir = Path(__file__).parent.parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent.parent / "Results" / "confusion_matrices"
results_dir.mkdir(parents=True, exist_ok=True)

model_path = experiments_dir / "rice_cnn_baseline_best.pth"

# ========== FUNCTIONS ==========
def plot_confusion_matrix(predictions, labels, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, square=True
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    output_path = results_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved: {output_path.name}")


def generate_classification_report(predictions, labels, model_name):
    """Generate and save classification report"""
    report = classification_report(labels, predictions, target_names=class_names, digits=4)

    print("\n" + "=" * 70)
    print(f"Classification Report - {model_name}")
    print("=" * 70)
    print(report)

    report_path = results_dir / f"classification_report_{model_name.replace(' ', '_').lower()}.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    print(f"[SUCCESS] Classification report saved: {report_path.name}")


# ========== EVALUATION ==========
print("\n" + "=" * 70)
print("BASELINE CNN EVALUATION")
print("=" * 70)

if not model_path.exists():
    print(f"[ERROR] Model not found at {model_path}")
    exit(1)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
print(f"[SUCCESS] Loaded model: {model_path.name}")

model.eval()
all_predictions = []
all_labels = []
correct = 0
total = 0

print("\nRunning inference on test set...")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")

accuracy = 100 * correct / total
print(f"\n[SUCCESS] Inference complete - Accuracy: {accuracy:.2f}%")

plot_confusion_matrix(all_predictions, all_labels, "Baseline CNN")
generate_classification_report(all_predictions, all_labels, "Baseline CNN")

print("\n" + "=" * 70)
