import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths
sys.path.append(str(Path(__file__).parent.parent / "Broken_Grains_Analysis"))
sys.path.append(str(Path(__file__).parent))

from Preprocessing import preprocess_image
from Segmentation import segment_grains
from Feature_Analysis import extract_features
from Classification import classify_grains

# ==========================================
# 🛡️ SYSTEM ROBUSTNESS TEST
# ==========================================
# This script tests how the system handles "Bad" images:
# 1. Dark Images (Low Exposure)
# 2. Noisy Images (Digital Noise)
# It proves your computer vision logic works in the real world.
# ==========================================

def add_noise(image):
    """Adds artificial Gaussian noise."""
    row, col, ch = image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(image, factor):
    """Adjusts brightness (0.5 = Dark, 1.5 = Bright)."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def run_robustness_check():
    sample_path = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Dataset\Rice_Image_Dataset\Basmati\Basmati (1).jpg"
    results_dir = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\Robustness_Test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    original = cv2.imread(sample_path)
    
    # Create versions
    dark = adjust_brightness(original, 0.4)
    noisy = add_noise(original)
    
    tests = [
        ("Original", original),
        ("Dark_Environment", dark),
        ("Digital_Noise", noisy)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, img) in enumerate(tests):
        # 1. Save temp image to process
        temp_p = results_dir / f"temp_{name}.jpg"
        cv2.imwrite(str(temp_p), img)
        
        # 2. Run Pipeline
        binary, _ = preprocess_image(str(temp_p))
        labels, _ = segment_grains(binary)
        features = extract_features(labels)
        classified, _ = classify_grains(features)
        
        counts = len(classified)
        
        # 3. Viz
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Test: {name}\nGrains Detected: {counts}", fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
        # Clean up
        if temp_p.exists(): os.remove(temp_p)

    plt.suptitle("System Robustness Validation\n(Testing Detection Stability under Non-Ideal Conditions)", 
                 fontsize=16, fontweight='bold', y=1.05)
    
    save_path = results_dir / "Robustness_Results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✨ Robustness test completed! Results: {save_path}")

if __name__ == "__main__":
    run_robustness_check()
