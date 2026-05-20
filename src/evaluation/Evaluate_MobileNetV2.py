import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ========== FIX PATH ==========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ========== IMPORTS ==========
from src.data.Dataset_loader import get_data_loaders
from torchvision import models


# ========== GET DATA ==========
_, _, test_loader, class_names = get_data_loaders(num_workers=0)


# ========== DEVICE ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ========== MODEL CONFIG ==========
NUM_CLASSES = len(class_names)
HIDDEN_UNITS = 512
DROPOUT_RATE = 0.5


# ========== LOAD MODEL ==========
print("\nLoading MobileNetV2 architecture...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Replace classifier (same as training)
last_channel = model.last_channel
model.classifier = nn.Sequential(
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(last_channel, HIDDEN_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(HIDDEN_UNITS, NUM_CLASSES)
)

model = model.to(device)


# ========== LOAD WEIGHTS ==========
experiments_dir = Path(__file__).parent.parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

model_path = experiments_dir / "rice_mobilenetv2_transfer_best.pth"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

print(f"[SUCCESS] Loaded trained MobileNetV2 model")


# ========== FUNCTIONS ==========

def evaluate_model():
    model.eval()

    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    print("\nEvaluating MobileNetV2...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy, all_predictions, all_labels


def plot_confusion_matrix(predictions, labels):
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix - MobileNetV2")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    output_path = results_dir / "confusion_matrix_mobilenetv2.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved: {output_path}")


def generate_classification_report(predictions, labels):
    report = classification_report(labels, predictions, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print("Classification Report - MobileNetV2")
    print("=" * 60)
    print(report)

    report_path = results_dir / "classification_report_mobilenetv2.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report - MobileNetV2\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print(f"[SUCCESS] Classification report saved: {report_path}")


# ========== MAIN ==========
print("=" * 60)
print("MOBILENETV2 EVALUATION")
print("=" * 60)

accuracy, preds, labels = evaluate_model()

if preds:
    plot_confusion_matrix(preds, labels)
    generate_classification_report(preds, labels)

print("=" * 60)