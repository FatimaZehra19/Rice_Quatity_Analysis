import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ==========================================
# ROBUSTNESS TEST: CV PIPELINE
# ==========================================
# Tests segmentation stability on degraded images
# ==========================================

def preprocess_image(image_path):
    """Convert image to binary mask."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)
    processed = cv2.medianBlur(thresh, 3)
    cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(processed, [c], 0, 255, -1)
    return processed, image


def segment_grains(binary_image):
    """Watershed segmentation."""
    binary_u8 = (binary_image > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = binary_u8.astype(bool)
    distance = ndimage.distance_transform_edt(binary)
    _, cc_labels = cv2.connectedComponents(binary_u8)
    n_components = int(np.max(cc_labels))
    if n_components == 0:
        return cc_labels.astype(int), distance
    component_areas = {
        lbl: int(np.sum(cc_labels == lbl))
        for lbl in range(1, n_components + 1)
        if int(np.sum(cc_labels == lbl)) >= 200
    }
    if not component_areas:
        return np.zeros_like(cc_labels, dtype=int), distance
    areas_sorted = sorted(component_areas.values())
    single_grain_area = areas_sorted[max(0, len(areas_sorted) // 4)]
    final_markers = np.zeros(distance.shape, dtype=int)
    next_label = 1
    for lbl, area in component_areas.items():
        comp_mask = (cc_labels == lbl)
        comp_dist = distance * comp_mask
        comp_max = float(np.max(comp_dist))
        if area / single_grain_area < 1.6:
            r, c = np.unravel_index(np.argmax(comp_dist), comp_dist.shape)
            final_markers[r, c] = next_label
            next_label += 1
        else:
            peaks = peak_local_max(
                comp_dist,
                min_distance=max(15, int(comp_max * 0.7)),
                threshold_abs=comp_max * 0.55,
                labels=comp_mask,
                footprint=np.ones((3, 3)),
                exclude_border=False,
            )
            if len(peaks) == 0:
                r, c = np.unravel_index(np.argmax(comp_dist), comp_dist.shape)
                final_markers[r, c] = next_label
            else:
                for (r, c) in peaks:
                    final_markers[r, c] = next_label
                    next_label += 1
    labels = watershed(-distance, final_markers, mask=binary)
    return labels, distance


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
    project_root = Path(__file__).parent.parent.parent
    sample_path = str(project_root / "Dataset" / "Rice_Image_Dataset" / "Basmati" / "Basmati (1).jpg")
    results_dir = project_root / "Results" / "Robustness_Test"
    results_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(sample_path)
    if original is None:
        print(f"❌ Could not find sample image at {sample_path}")
        return

    dark = adjust_brightness(original, 0.4)
    noisy = add_noise(original)

    tests = [("Original", original), ("Dark_Environment", dark), ("Digital_Noise", noisy)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (name, img) in enumerate(tests):
        temp_p = results_dir / f"temp_{name}.jpg"
        cv2.imwrite(str(temp_p), img)
        binary, _ = preprocess_image(str(temp_p))
        labels, _ = segment_grains(binary)
        grain_count = len(np.unique(labels)) - 1  # exclude background

        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{name}\nGrains: {grain_count}", fontsize=12, fontweight='bold')
        axes[i].axis('off')

        if temp_p.exists():
            os.remove(temp_p)

    plt.suptitle("Robustness Test: Segmentation Stability Under Degradation",
                 fontsize=16, fontweight='bold', y=1.05)
    save_path = results_dir / "Robustness_Results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Robustness test saved: {save_path}")


if __name__ == "__main__":
    run_robustness_check()
