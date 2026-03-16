# AI Coding Agent Instructions for Rice Classification Thesis

## Project Overview
This is a PyTorch-based deep learning project for multi-class **rice variety classification** (5 classes: Arborio, Basmati, Ipsala, Jasmine, Karacadag). The system compares baseline custom CNNs with transfer learning approaches (ResNet50, VGG16, MobileNet). Total dataset: 75,000 images (15,000 per class).

**Key Goal:** Develop accurate, efficient models for agricultural quality control with detailed performance comparisons.

## Architecture & Data Flow

### 1. Data Pipeline (Non-negotiable Convention)
- **Entry Point:** `src/Dataset_loader.py` - loads all data loaders for train/val/test
- **Image Size:** Fixed at 224×224 (ImageNet standard for transfer learning)
- **Normalization:** ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Split Ratio:** 70% train / 15% val / 15% test (via `random_split`)
- **Batch Size:** 32 (hardcoded in DataLoader calls)
- **Classes:** Always in sorted order: `['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']`

**When adding new models:** Always import `train_loader, val_loader, test_loader` from `Dataset_loader.py` - never recreate the pipeline.

### 2. Model Storage & Naming Convention
Models are saved to `Experiments/` with two patterns:
- **Best models:** `rice_{architecture}_best.pth` (e.g., `rice_resnet50_transfer_best.pth`)
- **Timestamped checkpoints:** `rice_{architecture}_{YYYYMMDD_HHMMSS}.pth`
- **Training histories:** `training_history_{architecture}_{timestamp}.json` (includes hyperparameters)

Load models via relative path: `Path(__file__).parent.parent / "Experiments" / model_name.pth`

### 3. Training Scripts Pattern
Two template models exist:
- **Baseline:** `Train.py` (custom CNN, trains from scratch)
- **Transfer Learning:** `Train.ResNet50.py` (ResNet50 + frozen backbone)

**Common structure for all train scripts:**
```python
# 1. Define hyperparameters as UPPERCASE constants at top
# 2. Device detection: Check CUDA → MPS → CPU (in that priority order, see Evaluate.py)
# 3. Training loop: model.train() / model.eval() phases with torch.no_grad()
# 4. Save best model when val_accuracy improves
# 5. Save training_history.json with all hyperparams and metrics
# 6. Print summary at end (use format from Train.ResNet50.py)
```

When implementing VGG16 or MobileNet: Follow ResNet50 template (frozen backbone + custom classifier head).

## Developer Workflows

### Training a New Architecture
```bash
# 1. Create Train.{Architecture}.py in src/
# 2. Copy structure from Train.ResNet50.py
# 3. Define hyperparameters at top (NUM_EPOCHS, LEARNING_RATE, etc.)
# 4. Load data: from Dataset_loader import train_loader, val_loader
# 5. Run: python src/Train.{Architecture}.py
# Output: saves model to Experiments/ + training_history JSON
```

### Evaluating Models
```bash
# Modify Evaluate.py to load your model
# Current setup only loads baseline CNN - need to generalize for all architectures
python src/Evaluate.py
# Output: confusion matrix PNG + classification report TXT in Results/
```

### Hyperparameter Tuning
All hyperparameters must be:
- Defined as UPPERCASE constants at file top
- Printed during initialization (see ResNet50 pattern)
- Included in saved `training_history.json`

Do not hardcode values mid-function; always read from module-level constants.

## Project Conventions & Patterns

### Reproducibility & Seed Management
All training scripts set `SEED = 42` at the start to ensure reproducible results:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```
This ensures:
- Same data shuffling order across runs
- Same random weight initialization
- Same dropout masks during training
- **Results are reproducible** - run the script twice, get identical accuracies

When adding new training scripts, always include this seed block before model initialization.

### Device Management
```python
# Preferred order (MacBook M1 support + CUDA fallback):
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

### Path Management (Always Use)
```python
from pathlib import Path
# Access files relative to project root:
experiments_dir = Path(__file__).parent.parent / "Experiments"
results_dir = Path(__file__).parent.parent / "Results"
experiments_dir.mkdir(parents=True, exist_ok=True)
```

### Metrics Tracking Pattern
```python
epoch_losses = []
val_accuracies = []
best_val_acc = 0.0
best_epoch = 0

# In training loop:
epoch_losses.append(loss_value)
val_accuracies.append(accuracy_value)
if accuracy_value > best_val_acc:
    best_val_acc = accuracy_value
    best_epoch = epoch + 1
    torch.save(model.state_dict(), best_model_path)
```

### Class Name Handling
Always use sorted class list (hardcoded in multiple files - standardize later):
```python
class_names = sorted(['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'])
# For Evaluate.py and visualization functions
```

## Known Issues & TODOs (Track These)

1. **Missing Test Evaluation:** Evaluate.py only tests baseline CNN (hardcoded `RiceCNN`). Need to generalize for ResNet50 and future models.
2. **No Early Stopping:** All models train full epochs without validation-based stopping. Recommend adding patience-based stopping.
3. **Hardcoded Class Names:** `Evaluate.py` line 37 hardcodes class list - should load from dataset metadata.
4. **Data Augmentation Missing:** Only basic transforms (resize, normalize). Add: rotation, color jitter, random crops.
5. **Incomplete Architectures:** VGG16 and MobileNet in README but not implemented yet.

## Before Modifying Code

- Always read the full file (not just snippets) - training scripts have interdependent patterns
- Check `src/Dataset_loader.py` first - it's the source of truth for data
- Check `Experiments/` for existing model artifacts and naming patterns
- Verify device detection matches `Evaluate.py` pattern (MPS priority for Mac)
- Ensure new scripts save to correct directories with correct naming

## Critical Integration Points

| Component | Purpose | Must Match |
|-----------|---------|-----------|
| Dataset_loader.py | Single source for all dataloaders | Image size (224×224), normalization |
| Models.py | Custom CNN class | Used by Train.py and Evaluate.py |
| Train.*.py | Training scripts | Save pattern, hyperparameter format |
| Evaluate.py | Test evaluation | Class names, device detection, model paths |
| Experiments/ | Model artifacts | Timestamped + best model pattern |

