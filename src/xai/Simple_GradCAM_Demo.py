import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(_PROJECT_ROOT / "src"))
sys.path.append(str(_PROJECT_ROOT / "src" / "models"))

from Baseline_CNN_Model import RiceCNN
from grad_cam import SimpleGradCAM

# ==========================================
# SETTINGS - CHANGE THESE TO TEST
# ==========================================
# Options: "Baseline", "MobileNetV2", "ResNet50"
MODEL_TYPE = "ResNet50"

# Choose ANY image from your dataset to test
IMAGE_PATH = str(_PROJECT_ROOT / "Dataset" / "Rice_Image_Dataset" / "Basmati" / "Basmati (1).jpg")
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ==========================================
# 🚀 3. RUN ANALYSIS
# ==========================================
def run_xai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = _PROJECT_ROOT / "Experiments"

    # LOAD CORRECT ARCHITECTURE
    if MODEL_TYPE == "Baseline":
        model = RiceCNN()
        target_layer = model.conv4
        weight_file = "rice_cnn_baseline_best.pth"
    elif MODEL_TYPE == "MobileNetV2":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(model.last_channel, 512),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 5)
        )
        target_layer = model.features[18][0]
        weight_file = "rice_mobilenetv2_transfer_best.pth"
    else: # ResNet50
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 5)
        )
        target_layer = model.layer4[-1]
        weight_file = "rice_resnet50_transfer_best.pth"

    # LOAD WEIGHTS
    weight_path = exp_dir / weight_file
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"✅ Loaded {MODEL_TYPE} weights!")
    
    model = model.to(device).eval()
    grad_cam = SimpleGradCAM(model, target_layer)

    # PREPARE IMAGE
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(Image.open(IMAGE_PATH).convert('RGB')).unsqueeze(0).to(device)

    # PROCESS
    heatmap, pred_idx = grad_cam.get_heatmap(input_tensor)

    # VISUALIZE
    original = cv2.imread(IMAGE_PATH)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    h_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    h_color = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.6, h_color, 0.4, 0)

    # SAVE PLOT
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(original); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(h_resized, cmap='jet'); plt.title("Heatmap"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title(f"XAI: {CLASS_NAMES[pred_idx]}"); plt.axis('off')
    
    save_path = _PROJECT_ROOT / "Results" / "XAI_Reports" / f"XAI_Single_Test_{MODEL_TYPE}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✨ Test Result saved to: {save_path}")

if __name__ == "__main__":
    run_xai()
