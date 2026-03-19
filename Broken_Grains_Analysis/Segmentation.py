import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def segment_grains(binary_image):
    """
    Improved watershed segmentation for overlapping grains
    (Balanced version – keeps peak detection but fixes merging issue)
    """

    # Ensure binary is correct type
    binary = binary_image.astype(bool)

    # Step 1: Distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Step 2: FIXED adaptive min_distance (VERY IMPORTANT)
    max_dist = np.max(distance)

    if max_dist > 0:
        # MUCH smaller than before (key fix)
        adaptive_min_dist = int(max_dist * 0.12)
        adaptive_min_dist = max(2, min(adaptive_min_dist, 10))
    else:
        adaptive_min_dist = 5

    # Step 3: Find local maxima (grain centers)
    local_maxi = peak_local_max(
        distance,
        min_distance=adaptive_min_dist,
        labels=binary,
        footprint=np.ones((3, 3))   # helps detect more peaks
    )

    # Step 4: Create markers
    markers = np.zeros(distance.shape, dtype=int)
    for i, (r, c) in enumerate(local_maxi):
        markers[r, c] = i + 1

    # Step 5: Apply watershed
    labels = watershed(-distance, markers, mask=binary)

    return labels, distance

if __name__ == "__main__":
    # Test block
    print("Segmentation script ready.")
