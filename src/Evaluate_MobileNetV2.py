import os
os.environ['TORCH_COMPILE_BACKEND'] = 'inductor'

import torch
import torch.nn as nn
from   pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# Import torchvision after torch
from torchvision import models

# Import dataset loader
from Dataset_loader import test_loader

# ========== DEVICE SETUP ==========
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ========== MODEL CONFIGURATION ==========
NUM_CLASSES = 5
HIDDEN_UNITS = 512
DROPOUT_RATE = 0.5

# Load the pretrained MobileNetV2 model
print("\n🔄 Loading pretrained MobileNet_V2 (ImageNet weights)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Replace the final classifier (same as training)
last_channel = model.last_channel
model.classifier = nn.Sequential(
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(last_channel, HIDDEN_UNITS),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(HIDDEN_UNITS, NUM_CLASSES)
)

model = model.to(device)
print("✓ Model architecture loaded")

# Load the best trained model weights
experiments_dir = Path(__file__).parent.parent / "Experiments"
best_model_path = experiments_dir / "rice_mobilenetv2_transfer_best.pth"

if not best_model_path.exists():
    print(f"❌ Error: Model not found at {best_model_path}")
    print("Please train the MobileNetV2 model first using Train.MobileNetV2.py")
    exit(1)

model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"✓ Best model loaded from: {best_model_path}")

# ========== EVALUATION SETUP ==========
class_names = sorted(['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'])
model.eval()

print("\n" + "="*70)
print("🧪 EVALUATING MOBILENETV2 MODEL ON TEST DATA")
print("="*70 + "\n")

# ========== TESTING LOOP ==========
all_predictions = []
all_labels = []
correct = 0
total = 0

print("Running inference on test set...\n")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Collect predictions and labels
        all_predictions.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        # Calculate accuracy
        total += labels.size(0)
        correct += torch.sum(predicted == labels).item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_acc = 100 * correct / total
            print(f"  Batch {batch_idx + 1}: Processed {total} images | Current Accuracy: {current_acc:.2f}%")

# ========== CALCULATE METRICS ==========
test_accuracy = 100 * correct / total

print(f"\n" + "="*70)
print("📊 TEST SET RESULTS")
print("="*70)
print(f"  Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
print("="*70 + "\n")

# ========== GENERATE CONFUSION MATRIX ==========
print("📈 Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_predictions)

# Create figure for confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette("forestgreen", as_cmap=True), 
            xticklabels=class_names, 
            yticklabels=class_names, 
            cbar_kws={'label': 'Count'},
            square=True)
plt.title('Confusion Matrix - MobileNetV2 Transfer Learning', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Save the confusion matrix as PNG
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(parents=True, exist_ok=True)

cm_path = results_dir / "confusion_matrix_mobilenetv2_model.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix PNG saved to: {cm_path}")
plt.close()

# ========== GENERATE CLASSIFICATION REPORT ==========
print("\n📋 Generating classification report...")

report = classification_report(all_labels, all_predictions, 
                              target_names=class_names, 
                              digits=4)

print(f"\n{'='*70}")
print("Classification Report - MobileNetV2 Transfer Learning")
print(f"{'='*70}")
print(report)

# Save classification report to file
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(parents=True, exist_ok=True)

report_path = results_dir / "classification_report_mobilenetv2_model.txt"
with open(report_path, 'w') as f:
    f.write("MobileNetV2 Transfer Learning - Classification Report\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})\n")
    f.write(f"Total Test Samples: {total}\n")
    f.write(f"Correct Predictions: {correct}\n")
    f.write(f"Incorrect Predictions: {total - correct}\n\n")
    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write("="*70 + "\n")
    f.write(report)

print(f"✓ Classification report saved to: {report_path}")

# ========== FINAL SUMMARY ==========
print("\n" + "="*70)
print("✅ EVALUATION COMPLETED!")
print("="*70)
print("\n💾 Saved Results:")
print(f"  • Confusion Matrix: confusion_matrix_mobilenetv2_model.png")
print(f"  • Classification Report: classification_report_mobilenetv2_model.txt")
print("="*70)
