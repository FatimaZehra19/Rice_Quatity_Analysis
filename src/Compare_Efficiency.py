import torch
import torch.nn as nn
import time
import os
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add src to path to load Baseline
sys.path.append(str(Path(__file__).parent))
from Baseline_CNN_Model import RiceCNN

# ==========================================
# 🚀 PERFORMANCE BENCHMARKING TOOL
# ==========================================
# This script measures:
# 1. Model Size (MB) - Critical for mobile/edge deployment.
# 2. Inference Speed (ms) - Time taken to predict one image.
# ==========================================

def get_model_size(file_path):
    """Returns the size of the weight file in Megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def measure_inference_speed(model, device, iterations=50):
    """Measures average time for one forward pass (prediction)."""
    # Create a dummy image (224x224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warm-up (important for accurate GPU/CPU timing)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    avg_time = (time.time() - start_time) / iterations
    return avg_time * 1000  # Convert to milliseconds

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Experiments")
    
    models_to_test = ["Baseline", "MobileNetV2", "ResNet50"]
    results = {}

    print(f"🖥️  Starting Benchmark on {device}...")

    for m_type in models_to_test:
        print(f"📊 Benchmarking {m_type}...")
        
        # 1. Initialize Architecture
        if m_type == "Baseline":
            model = RiceCNN()
            weight_file = "rice_cnn_baseline_best.pth"
        elif m_type == "MobileNetV2":
            model = models.mobilenet_v2()
            model.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(model.last_channel, 512),
                nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 5)
            )
            weight_file = "rice_mobilenetv2_transfer_best.pth"
        else:
            model = models.resnet50()
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 5)
            )
            weight_file = "rice_resnet50_transfer_best.pth"

        weight_path = exp_dir / weight_file
        
        if weight_path.exists():
            # 2. Measure Size
            size_mb = get_model_size(weight_path)
            
            # 3. Measure Speed
            model.to(device).eval()
            speed_ms = measure_inference_speed(model, device)
            
            results[m_type] = {"Size (MB)": round(size_mb, 2), "Inference (ms)": round(speed_ms, 2)}
        else:
            print(f"⚠️  Skipping {m_type}: Weights not found.")

    # ==========================================
    # 📈 PLOTTING RESULTS
    # ==========================================
    names = list(results.keys())
    sizes = [r["Size (MB)"] for r in results.values()]
    speeds = [r["Inference (ms)"] for r in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Size
    colors = ['#2196F3', '#FF9800', '#F44336']
    bars1 = ax1.bar(names, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Model Footprint (File Size)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Megabytes (MB)", fontsize=12)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{bar.get_height()} MB', ha='center', fontweight='bold')

    # Plot 2: Speed
    bars2 = ax2.bar(names, speeds, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title("Inference Lag (Predict Time)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Milliseconds (ms) per Image", fontsize=12)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height()} ms', ha='center', fontweight='bold')

    plt.suptitle(f"Computational Efficiency Analysis ({device})", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    save_path = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\Efficiency_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✨ Benchmark Results saved to: {save_path}")

if __name__ == "__main__":
    run_benchmark()
