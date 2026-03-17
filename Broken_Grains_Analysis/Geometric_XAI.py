import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from Preprocessing import preprocess_image
from Segmentation import segment_grains
from Feature_Analysis import extract_features
from Classification import classify_grains

# ==============================================================================
# 📏 GEOMETRIC XAI: EXPLAINING THE 'MATH' OF GRAIN QUALITY
# ==============================================================================
# This script explains the decisions made by the non-AI (Computer Vision)
# pipeline. It audits why a grain is marked 'Broken' versus 'Full'.
# ------------------------------------------------------------------------------

def visualize_geometric_logic_with_reason(image_path, output_path):
    """Generates a Visual Audit showing the reasoning for every grain."""
    
    # 1. Image Base Name and Setup
    image_name = os.path.basename(image_path)
    
    # 2. RUN THE PIPELINE (Process the grain)
    binary, original = preprocess_image(image_path)
    if binary is None: return
    labels, _ = segment_grains(binary)
    features = extract_features(labels)
    
    # 3. GET CLASSIFICATION (And retrieve the Reference/Max values)
    # The system considers the LARGEST grain in the image as the 'Standard'
    classified_data, (max_len, max_area) = classify_grains(features)
    
    # Thresholds: Anything below these is considered damaged/broken
    len_limit = max_len * 0.75  # Must be at least 75% of max length
    area_limit = max_area * 0.70 # Must be at least 70% of max area
    
    # 4. CREATE VISUALIZATION (Comparison View)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- AX 1: THE VISUAl AUDIT (The Image) ---
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    for i, g in enumerate(classified_data):
        color = 'lime' if g['classification'] == 'Full' else 'red'
        ax1.plot(g['centroid'][0], g['centroid'][1], 'o', color=color, markersize=5)
        
        # Add labels explaining the "WHY"
        if g['classification'] == 'Broken':
            reasons = []
            if g['length'] < len_limit: reasons.append(f"Len:{g['length']:.0f}<{len_limit:.0f}")
            if g['area'] < area_limit: reasons.append(f"Area:{g['area']:.0f}<{area_limit:.0f}")
            reason_text = " & ".join(reasons)
            ax1.text(g['centroid'][0], g['centroid'][1] + 20, reason_text, 
                    color='white', fontsize=7, backgroundcolor='red', weight='bold')
        else:
            ax1.text(g['centroid'][0], g['centroid'][1] + 20, "PASS: FULL", 
                    color='white', fontsize=7, backgroundcolor='green', weight='bold')

    ax1.set_title(f"Visual Reason Audit: {image_name}\n(White labels show failing metrics)", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # --- AX 2: THE FEATURE SPACE (The Logic Map) ---
    lengths = [g['length'] for g in classified_data]
    areas = [g['area'] for g in classified_data]
    colors = ['green' if g['classification'] == 'Full' else 'red' for g in classified_data]
    
    ax2.scatter(lengths, areas, c=colors, edgecolors='black', s=100, alpha=0.7)
    
    # Draw the "Decision Boundaries" (The lines that separate Full from Broken)
    ax2.axvline(x=len_limit, color='red', linestyle='--', label=f'Min Length ({len_limit:.1f})')
    ax2.axhline(y=area_limit, color='blue', linestyle='--', label=f'Min Area ({area_limit:.1f})')
    
    ax2.set_xlabel('Grain Length (Pixels)', fontsize=12)
    ax2.set_ylabel('Grain Area (Pixels)', fontsize=12)
    ax2.set_title("Decision Boundary Logic Map\n(Area and Length Constraints)", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 5. SAVE AND FINISH
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"EXPLAINED_logic_{image_name}.png")
    plt.savefig(save_path, dpi=300)
    
    print("-" * 50)
    print(f"✅ 'Why' Explanation saved to:\n   {save_path}")
    print("-" * 50)

if __name__ == "__main__":
    # Test on a Basmati sample image
    sample_img = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Dataset\Rice_Image_Dataset\Basmati\Basmati (1).jpg"
    out_dir = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\XAI_Reports"
    visualize_geometric_logic_with_reason(sample_img, out_dir)
