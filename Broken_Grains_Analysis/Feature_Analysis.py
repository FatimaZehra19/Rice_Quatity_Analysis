import cv2
import numpy as np

def extract_features(labels):
    """
    Extract geometric features from segmented grains
    with area-based filtering to remove noise and over-segmentation.
    """

    grain_features = []

    # 🔥 Thresholds (tune if needed)
    MIN_AREA = 25     # remove very small noise
    MAX_AREA = 12000  # optional: remove very large merged grains

    # Iterate through each grain
    for label in np.unique(labels):
        if label == 0:
            continue

        # Create mask
        mask = np.uint8(labels == label) * 255

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        cnt = contours[0]

        # ---------------------------
        # 1. AREA
        # ---------------------------
        area = cv2.contourArea(cnt)

        # 🚨 AREA FILTER (IMPORTANT)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # ---------------------------
        # 2. AXIS LENGTHS
        # ---------------------------
        if len(cnt) >= 5:
            (x, y), (d1, d2), angle = cv2.fitEllipse(cnt)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            major_axis = max(w, h)
            minor_axis = min(w, h)
            x, y = int(x + w / 2), int(y + h / 2)

        # ---------------------------
        # 3. ASPECT RATIO
        # ---------------------------
        aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0

        # ---------------------------
        # STORE FEATURES
        # ---------------------------
        grain_features.append({
            'label': int(label),
            'area': float(area),
            'length': float(major_axis),
            'width': float(minor_axis),
            'aspect_ratio': float(aspect_ratio),
            'centroid': (int(x), int(y))
        })

    return grain_features