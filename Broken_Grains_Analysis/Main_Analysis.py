import cv2
import os
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys

# Import our custom modules
from Preprocessing import preprocess_image
from Segmentation import segment_grains
from Feature_Analysis import extract_features
from Classification import classify_grains

# Add src to path so we can load the model architecture
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
from Baseline_CNN_Model import RiceCNN

# ==========================================
# 🧠 XAI GRAD-CAM ENGINE
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
# 🚀 ANALYSIS CORE
# ==========================================

def analyze_rice_sample(image_path, model, grad_cam, save_image=False, output_folder=None):
    """
    Runs complete analysis: Preprocessing -> Segmentation -> Feature Analysis -> XAI Heatmap
    """
    image_name = os.path.basename(image_path)
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    # 1. Computer Vision Pipeline (Broken Grain Detection)
    binary, original = preprocess_image(image_path)
    if binary is None: return None
    labels, _ = segment_grains(binary)
    features = extract_features(labels)
    classified_data, (max_len, max_area) = classify_grains(features)
    
    full_count = sum(1 for g in classified_data if g['classification'] == 'Full')
    broken_count = len(classified_data) - full_count
    total = len(classified_data)
    percent_broken = (broken_count / total * 100) if total > 0 else 0
    
    # 2. XAI / Deep Learning Pipeline (Variety Confirmation) 
    if save_image and model and grad_cam:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(next(model.parameters()).device)
        heatmap, pred_idx = grad_cam.get_heatmap(input_tensor)
        
        # Superimpose Heatmap
        h_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        h_color = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), 0.6, h_color, 0.4, 0)

        # Draw Labels on Original for Broken vs Full
        label_viz = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        for g in classified_data:
            color = (0, 255, 0) if g['classification'] == 'Full' else (255, 0, 0)
            cv2.circle(label_viz, g['centroid'], 5, color, -1)

        # CREATE SIDE-BY-SIDE REPORT
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(label_viz)
        plt.title(f"Broken Grain Detection\n(Broken: {broken_count}, Full: {full_count})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"XAI Variety Insight: {class_names[pred_idx]}\n(Model Confidence Highlighted)")
        plt.axis('off')

        os.makedirs(output_folder, exist_ok=True)
        report_path = os.path.join(output_folder, f"quality_report_{image_name}.png")
        plt.savefig(report_path, dpi=200, bbox_inches='tight')
        plt.close()

    return {
        "Image": image_name,
        "Total": total,
        "Full": full_count,
        "Broken": broken_count,
        "% Broken": round(percent_broken, 2)
    }

def main():
    # SETTINGS
    dataset_root = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Dataset\Rice_Image_Dataset"
    output_path = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\Final_Broken_Grain_Report"
    exp_dir = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Experiments"
    
    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Analysis (Device: {device})...")
    
    # Load Baseline Model for XAI (User preferred Baseline heatmap)
    model = RiceCNN()
    weight_path = os.path.join(exp_dir, "rice_cnn_baseline_best.pth")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device).eval()
        grad_cam = SimpleGradCAM(model, model.conv4)
        print("✓ XAI Model Loaded (Baseline CNN)")
    else:
        model, grad_cam = None, None
        print("⚠️ XAI Model weights not found. Skipping heatmaps.")

    categories = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    summary_data = {}

    for category in categories:
        print(f"📂 Processing category: {category}...")
        cat_path = os.path.join(dataset_root, category)
        images = glob.glob(os.path.join(cat_path, "*.jpg"))[:50] # 50 samples
        
        cat_results = []
        for i, img_path in enumerate(images):
            # Save only the FIRST image as a visual Quality Report
            save_this = (i == 0)
            data = analyze_rice_sample(img_path, model, grad_cam, save_image=save_this, output_folder=output_path)
            if data: cat_results.append(data)
        
        avg_broken = sum(d['% Broken'] for d in cat_results) / len(cat_results) if cat_results else 0
        summary_data[category] = {"avg_broken": round(avg_broken, 2), "images": len(cat_results)}
        print(f"   ✓ {category} complete: {round(avg_broken, 2)}% broken")

    # Save final report
    json_path = os.path.join(output_path, "quality_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"\n✨ Analysis Completed!")
    print(f"📁 Detailed Quality Reports saved in: {output_path}")

if __name__ == "__main__":
    main()
