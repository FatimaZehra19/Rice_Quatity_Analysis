import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ========== SETUP ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from Baseline_CNN_Model import RiceCNN

# ========== CONSTANTS ==========
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

print(f"Using device: {DEVICE}")
print("=" * 70)

# ========== LOAD TEST DATA ==========
dataset_path = PROJECT_ROOT / "Dataset" / "Rice_Image_Dataset"

if not dataset_path.exists():
    print(f"ERROR: Dataset not found at {dataset_path}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=str(dataset_path), transform=transform)

# Get test indices (last 15% of dataset)
dataset_size = len(full_dataset)
test_size = int(0.15 * dataset_size)
train_size = int(0.7 * dataset_size)
val_size = dataset_size - train_size - test_size

# Use indices to get test set
torch.manual_seed(42)
from torch.utils.data import random_split
_, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# DataLoader with num_workers=0 to avoid Windows multiprocessing issues
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[SUCCESS] Dataset loaded: {len(test_dataset)} test images")
print(f"[SUCCESS] Test DataLoader created with batch_size={BATCH_SIZE}")
print("=" * 70)

# ========== EVALUATION FUNCTION ==========
def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions"""
    model.eval()
    all_predictions = []
    all_labels = []

    print("Running inference on test set...")

    with torch.no_grad():
        batch_count = 0
        total_batches = len(test_loader)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_count += 1
            if batch_count % 50 == 0:
                print(f"  Processed batch {batch_count}/{total_batches}")

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_predictions == all_labels) * 100

    print(f"[SUCCESS] Inference complete - Accuracy: {accuracy:.2f}%")

    return all_predictions, all_labels, accuracy

# ========== GENERATE CONFUSION MATRIX ==========
def generate_confusion_matrix(predictions, labels, model_name, output_dir):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'},
        square=True
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved: {output_path}")

# ========== GENERATE CLASSIFICATION REPORT ==========
def generate_classification_report(predictions, labels, model_name, output_dir):
    """Generate and save classification report"""
    report = classification_report(labels, predictions, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 70)
    print(f"Classification Report - {model_name}")
    print("=" * 70)
    print(report)

    report_path = output_dir / f"classification_report_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)

    print(f"[SUCCESS] Classification report saved: {report_path}")

# ========== MAIN EVALUATION ==========
results_dir = PROJECT_ROOT / "Results" / "confusion_matrices"
results_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("EVALUATING ALL MODELS")
print("=" * 70)

# ========== MODEL 1: BASELINE CNN ==========
print("\n[1/3] BASELINE CNN")
print("-" * 70)

baseline_model = RiceCNN().to(DEVICE)
baseline_path = PROJECT_ROOT / "Experiments" / "rice_cnn_baseline_best.pth"

if not baseline_path.exists():
    print(f"[ERROR] Model not found at {baseline_path}")
else:
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    print(f"[SUCCESS] Loaded model from {baseline_path}")

    baseline_preds, baseline_labels, baseline_acc = evaluate_model(baseline_model, test_loader, DEVICE)
    generate_confusion_matrix(baseline_preds, baseline_labels, "Baseline CNN", results_dir)
    generate_classification_report(baseline_preds, baseline_labels, "Baseline CNN", results_dir)

# ========== MODEL 2: MOBILENETV2 ==========
print("\n[2/3] MOBILENETV2")
print("-" * 70)

mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
last_channel = mobilenet_model.last_channel
mobilenet_model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(last_channel, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
mobilenet_model = mobilenet_model.to(DEVICE)

mobilenet_path = PROJECT_ROOT / "Experiments" / "rice_mobilenetv2_transfer_best.pth"

if not mobilenet_path.exists():
    print(f"[ERROR] Model not found at {mobilenet_path}")
else:
    mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location=DEVICE))
    mobilenet_model = mobilenet_model.to(DEVICE)
    print(f"[SUCCESS] Loaded model from {mobilenet_path}")

    mobilenet_preds, mobilenet_labels, mobilenet_acc = evaluate_model(mobilenet_model, test_loader, DEVICE)
    generate_confusion_matrix(mobilenet_preds, mobilenet_labels, "MobileNetV2", results_dir)
    generate_classification_report(mobilenet_preds, mobilenet_labels, "MobileNetV2", results_dir)

# ========== MODEL 3: RESNET50 ==========
print("\n[3/3] RESNET50")
print("-" * 70)

resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Sequential(
    nn.Linear(resnet_model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
resnet_model = resnet_model.to(DEVICE)

resnet_path = PROJECT_ROOT / "Experiments" / "rice_resnet50_transfer_best.pth"

if not resnet_path.exists():
    print(f"[ERROR] Model not found at {resnet_path}")
else:
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
    resnet_model = resnet_model.to(DEVICE)
    print(f"[SUCCESS] Loaded model from {resnet_path}")

    resnet_preds, resnet_labels, resnet_acc = evaluate_model(resnet_model, test_loader, DEVICE)
    generate_confusion_matrix(resnet_preds, resnet_labels, "ResNet50", results_dir)
    generate_classification_report(resnet_preds, resnet_labels, "ResNet50", results_dir)

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("ALL CONFUSION MATRICES GENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nResults saved to: {results_dir}")
print("\nFiles created:")
print("  - confusion_matrix_baseline_cnn.png")
print("  - confusion_matrix_mobilenetv2.png")
print("  - confusion_matrix_resnet50.png")
print("  - classification_report_baseline_cnn.txt")
print("  - classification_report_mobilenetv2.txt")
print("  - classification_report_resnet50.txt")
print("=" * 70)
