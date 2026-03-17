import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"⚠ Warning: Unable to read image at {image_path}")
        return None, None
    
    # 1. Convert to grayscale (image, not image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Applying Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3. Applying Otsu Thresholding to get binary image
    # Note: Using THRESH_BINARY as the dataset usually has white grains on dark/black backgrounds.
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Cleanup: Morphological Opening to remove small noise
    kernel = np.ones((3,3), np.uint8)
    processed_binary = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return processed_binary, image