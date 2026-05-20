import sys
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Python path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Imports
from src.data.Dataset_loader import get_data_loaders
from src.models.Baseline_CNN_Model import RiceCNN


# ========== GET DATA ==========
_, _, test_loader, class_names = get_data_loaders(num_workers=0)


# ========== DEVICE SETUP ==========
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ========== MODEL SETUP ==========
model = RiceCNN().to(device)


# ========== PATH SETUP ==========
experiments_dir = Path(__file__).parent.parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

best_model_path = experiments_dir / "rice_cnn_baseline_best.pth"

# Find latest baseline model (excluding best)
baseline_models = [
    m for m in experiments_dir.glob("rice_cnn_baseline_*.pth")
    if m.name != "rice_cnn_baseline_best.pth"
]
baseline_model_path = sorted(baseline_models)[-1] if baseline_models else None


# ========== FUNCTIONS ==========

def plot_confusion_matrix(predictions, labels, model_name):
    """Plot and save confusion matrix"""
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
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    output_path = results_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved: {output_path}")


def generate_classification_report(predictions, labels, model_name):
    """Generate and save classification report"""
    report = classification_report(labels, predictions, target_names=class_names, digits=4)

    print("\n" + "=" * 60)
    print(f"Classification Report - {model_name}")
    print("=" * 60)
    print(report)

    report_path = results_dir / f"classification_report_{model_name.replace(' ', '_').lower()}.txt"
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    print(f"[SUCCESS] Classification report saved: {report_path}")


def evaluate_model(model_path, model_name):
    """Evaluate model on test set"""
    if not model_path or not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return None, None, None

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"\n[SUCCESS] Loaded {model_name}: {model_path.name}")

    model.eval()

    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    print(f"Evaluating {model_name}...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * test_correct / test_total

    print(f"Total Samples: {test_total}")
    print(f"Correct Predictions: {test_correct}")
    print(f"{model_name} Accuracy: {accuracy:.2f}%")

    return accuracy, all_predictions, all_labels


# ========== MAIN EXECUTION ==========

print("=" * 60)
print("TEST EVALUATION")
print("=" * 60)


# Evaluate best model
if best_model_path.exists():
    acc, preds, labels = evaluate_model(best_model_path, "Best Model")
    if preds:
        plot_confusion_matrix(preds, labels, "Best Model")
        generate_classification_report(preds, labels, "Best Model")


# Evaluate baseline model
if baseline_model_path:
    acc, preds, labels = evaluate_model(baseline_model_path, "Baseline Model")
    if preds:
        plot_confusion_matrix(preds, labels, "Baseline Model")
        generate_classification_report(preds, labels, "Baseline Model")


print("=" * 60)