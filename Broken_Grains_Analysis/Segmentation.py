import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def segment_grains(binary_image):
    """
    Applies the Watershed algorithm to segment individual grains.
    """
    # 1. Distance transform: find the distance to the nearest background pixel for each foreground pixel
    distance = ndimage.distance_transform_edt(binary_image)
    
    # 2. Local maximums: find the centers of the grains
    # min_distance determines how close grains can be before being merged
    local_maxi = peak_local_max(distance, min_distance=50, labels=binary_image)
    
    # 3. Create markers for watershed
    markers = np.zeros(distance.shape, dtype=int)
    for i, (r, c) in enumerate(local_maxi):
        markers[r, c] = i + 1
        
    # 4. Apply Watershed
    labels = watershed(-distance, markers, mask=binary_image)
    
    return labels, distance

if __name__ == "__main__":
    # Test block
    print("Segmentation script ready.")
