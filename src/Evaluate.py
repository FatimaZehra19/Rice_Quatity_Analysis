import torch
import torch.nn as nn
from   pathlib import Path
from   Dataset_loader import test_loader
from   Models import RiceCNN
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define the device — prefer Apple Silicon GPU (MPS) over CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize the model
model = RiceCNN()
model = model.to(device)

# Load the saved trained model
experiments_dir = Path(__file__).parent.parent / "Experiments"
best_model_path = experiments_dir / "rice_cnn_baseline_best.pth"

# Find baseline model (latest timestamped model)
baseline_models = list(experiments_dir.glob("rice_cnn_baseline_*.pth"))
baseline_model_path = None
if baseline_models:
    # Filter out the best model and get the latest timestamped one
    baseline_models = [m for m in baseline_models if m.name != "rice_cnn_baseline_best.pth"]
    if baseline_models:
        baseline_model_path = sorted(baseline_models)[-1]  # Get the latest one

def plot_confusion_matrix(predictions, labels, model_name):
    """Plot and save confusion matrix as PNG"""
    # Get class names from the dataset
    try:
        # Assuming class names correspond to dataset folder names
        class_names = sorted(['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'])
    except:
        class_names = None
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette("steelblue", as_cmap=True), xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Create Results folder if it doesn't exist
    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Save the figure
    output_path = results_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()
    
    return output_path


def generate_classification_report(predictions, labels, model_name):
    """Generate and save classification report with precision, recall, and f1-score"""
    # Get class names
    class_names = sorted(['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'])
    
    # Generate classification report
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    
    print(f"\n{'='*60}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*60}")
    print(report)
    
    # Save report to file
    results_dir = Path(__file__).parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    
    report_path = results_dir / f"classification_report_{model_name.replace(' ', '_').lower()}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(report)
    
    print(f"✓ Classification report saved to: {report_path}")
    
    return report_path


def evaluate_model(model_path, model_name):
    """Evaluate a model on the test set"""
    if not model_path.exists():
        print(f"✗ Model file not found at: {model_path}")
        return None, None, None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\n✓ Loaded {model_name}: {model_path.name}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate Test Accuracy
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    print(f"Evaluating {model_name} on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Count correct predictions
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and print test accuracy
    test_accuracy = 100 * test_correct / test_total
    print(f"Total Test Samples: {test_total}")
    print(f"Correct Predictions: {test_correct}")
    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy, all_predictions, all_labels

# Evaluate models
print("="*60)
print("Test Evaluation Results")
print("="*60)

# Evaluate best model
if best_model_path.exists():
    best_accuracy, best_predictions, best_labels = evaluate_model(best_model_path, "Best Model")
    if best_predictions is not None:
        plot_confusion_matrix(best_predictions, best_labels, "Best Model")
        generate_classification_report(best_predictions, best_labels, "Best Model")

# Evaluate baseline model
if baseline_model_path:
    baseline_accuracy, baseline_predictions, baseline_labels = evaluate_model(baseline_model_path, "Baseline Model")
    if baseline_predictions is not None:
        plot_confusion_matrix(baseline_predictions, baseline_labels, "Baseline Model")
        generate_classification_report(baseline_predictions, baseline_labels, "Baseline Model")

print("="*60)
