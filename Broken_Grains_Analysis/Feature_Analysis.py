import cv2
import numpy as np

def extract_features(labels):
    """
    Analyzes each segmented grain and extracts geometric features.
    """
    grain_features = []
    
    # Iterate through each unique labeled grain (skipping background 0)
    for label in np.unique(labels):
        if label == 0:
            continue
            
        # Create a mask for the current grain
        mask = np.uint8(labels == label) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        cnt = contours[0]
        
        # 1. Area
        area = cv2.contourArea(cnt)
        
        # 2. Major and Minor Axis Length using FitEllipse
        if len(cnt) >= 5: # FitEllipse needs at least 5 points
            (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(cnt)
        else:
            # Fallback for very small grains
            x, y, w, h = cv2.boundingRect(cnt)
            major_axis = max(w, h)
            minor_axis = min(w, h)
        
        # 3. Aspect Ratio
        aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
        
        grain_features.append({
            'label': int(label),
            'area': area,
            'length': major_axis,
            'width': minor_axis,
            'aspect_ratio': aspect_ratio,
            'centroid': (int(x), int(y))
        })
        
    return grain_features
