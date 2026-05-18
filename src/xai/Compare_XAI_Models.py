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

# ------------------------------------------------------------------------------
# 🏗️ MODEL LOADER
# Loads the specific architecture and trained weights for each model type.
# ------------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent

def create_model(model_type, device):
    """Initializes the model architecture and loads your trained .pth weights."""
    exp_dir = _PROJECT_ROOT / "Experiments"
    
    if model_type == "Baseline":
        model = RiceCNN()
        target_layer = model.conv4 # The final convolutional layer
        weight_file = "rice_cnn_baseline_best.pth"
    elif model_type == "MobileNetV2":
        model = models.mobilenet_v2()
        # Custom head added during training
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
        target_layer = model.features[18][0] # Depthwise conv layer
        weight_file = "rice_mobilenetv2_transfer_best.pth"
    else: # ResNet50
        model = models.resnet50()
        # Custom head added during training
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
        target_layer = model.layer4[-1] # Final bottleneck block
        weight_file = "rice_resnet50_transfer_best.pth"

    # Load the actual trained weights from your Experiments folder
    weight_path = exp_dir / weight_file
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"✅ {model_type} weights loaded successfully.")
    
    return model.to(device).eval(), target_layer

# ------------------------------------------------------------------------------
# 🚀 MAIN COMPARISON RUNNER
# ------------------------------------------------------------------------------
def run_triple_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Change this to test a different image
    img_path = str(_PROJECT_ROOT / "Dataset" / "Rice_Image_Dataset" / "Basmati" / "Basmati (1).jpg")
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    # Standard preprocessing used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process image for the model
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Load original image for plotting
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))
    
    # Run analysis for all 3 models
    models_to_test = ["Baseline", "MobileNetV2", "ResNet50"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # First slot: The Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Grain", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for i, m_type in enumerate(models_to_test):
        # 1. Load the model
        model, target_layer = create_model(m_type, device)
        grad_cam = SimpleGradCAM(model, target_layer)
        
        # 2. Get the heatmap
        heatmap, pred_idx = grad_cam.get_heatmap(input_tensor)
        
        # 3. Superimpose the 'Heatmap' onto the Original Image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        
        # 4. Show the result
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"{m_type}\nChoice: {class_names[pred_idx]}", fontsize=12, fontweight='bold')
        axes[i+1].axis('off')

    plt.suptitle(f"XAI Comparison: How different models interpret {Path(img_path).stem}", 
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save the professional result
    res_dir = _PROJECT_ROOT / "Results" / "XAI_Reports"
    res_dir.mkdir(parents=True, exist_ok=True)
    save_path = res_dir / "XAI_DeepLearning_Comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 50)
    print(f"✨ COMPLETED! XAI Comparison saved to:\n   {save_path}")
    print("-" * 50)

if __name__ == "__main__":
    run_triple_comparison()
