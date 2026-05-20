import sys
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "data"))
from Dataset_loader import get_data_loaders

# ========== SETUP ==========
_, _, test_loader, _ = get_data_loaders(batch_size=32, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")

NUM_CLASSES = 5
HIDDEN_UNITS = 512
DROPOUT_RATE = 0.5
CLASS_NAMES = sorted(['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'])

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, HIDDEN_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(HIDDEN_UNITS, NUM_CLASSES)
)
model = model.to(device)

experiments_dir = Path(__file__).parent.parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent.parent / "Results" / "confusion_matrices"
results_dir.mkdir(parents=True, exist_ok=True)

best_model_path = experiments_dir / "rice_resnet50_transfer_best.pth"

# ========== FUNCTIONS ==========
def plot_confusion_matrix(predictions, labels):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}, square=True
    )
    plt.title('Confusion Matrix - ResNet50', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    cm_path = results_dir / "confusion_matrix_resnet50.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved: {cm_path.name}")


def generate_classification_report(predictions, labels):
    """Generate and save classification report"""
    report = classification_report(labels, predictions, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 70)
    print("Classification Report - ResNet50")
    print("=" * 70)
    print(report)

    report_path = results_dir / "classification_report_resnet50.txt"
    with open(report_path, 'w') as f:
        f.write("ResNet50 Transfer Learning - Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    print(f"[SUCCESS] Classification report saved: {report_path.name}")


# ========== EVALUATION ==========
print("\n" + "=" * 70)
print("RESNET50 EVALUATION")
print("=" * 70)

if not best_model_path.exists():
    print(f"[ERROR] Model not found at {best_model_path}")
    exit(1)

model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
print(f"[SUCCESS] Loaded model: {best_model_path.name}")

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

test_accuracy = 100 * correct / total
print(f"\n[SUCCESS] Inference complete - Accuracy: {test_accuracy:.2f}%")

plot_confusion_matrix(all_predictions, all_labels)
generate_classification_report(all_predictions, all_labels)

print("\n" + "=" * 70)
