import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from Baseline_CNN_Model import RiceCNN

# Use 'Agg' to avoid terminal freezing, but it still saves the image!
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 🏠 1. SETTINGS - CHANGE THESE TO TEST
# ==========================================
# Options: "Baseline", "MobileNetV2", "ResNet50"
MODEL_TYPE = "ResNet50" 

# Choose ANY image from your dataset to test
IMAGE_PATH = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Dataset\Rice_Image_Dataset\Basmati\Basmati (1).jpg"
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# ==========================================
# 🧠 2. GRAD-CAM ENGINE
# ==========================================
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= (torch.max(heatmap) + 1e-10)
        return heatmap.detach().cpu().numpy(), idx

# ==========================================
# 🚀 3. RUN ANALYSIS
# ==========================================
def run_xai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Experiments")

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
    
    save_path = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\XAI_Reports") / f"XAI_Single_Test_{MODEL_TYPE}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✨ Test Result saved to: {save_path}")

if __name__ == "__main__":
    run_xai()
