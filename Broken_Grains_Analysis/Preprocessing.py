import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"⚠ Warning: Unable to read image at {image_path}")
        return None, None
    
    # 1. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Applying Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3. Applying Otsu Thresholding to get binary image
    # Note: We check if the background is light or dark
    # Most rice datasets have dark backgrounds, but real photos might vary.
    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Simple check: if more than 50% pixels are white, invert (assuming background should be black)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    # 4. Cleanup: Simple noise removal while preserving small particles
    # Instead of opening, we use a small median blur to remove single-pixel noise
    processed_binary = cv2.medianBlur(thresh, 3)
    
    # 5. Fill holes in grains
    cnts, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(processed_binary, [c], 0, 255, -1)
    
    return processed_binary, image