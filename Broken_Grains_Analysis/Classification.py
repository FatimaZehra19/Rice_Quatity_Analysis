import numpy as np

def classify_grains(grain_features):
    """
    This function decides if a grain is 'Full' or 'Broken'.
    It compares each grain to the largest grain found in the image.
    """
    
    # If there are no grains to analyze, just stop
    if len(grain_features) == 0:
        return [], (0, 0)
    
    # STEP 1: Collect all lengths and areas to find the biggest one
    all_lengths = []
    all_areas = []
    
    for grain in grain_features:
        all_lengths.append(grain['length'])
        all_areas.append(grain['area'])
    
    # STEP 2: Find the 'Reference' size (the Max size in this image)
    # We assume the largest grain found is a 'Full Grain'
    max_length = max(all_lengths)
    max_area = max(all_areas)
    
    # STEP 3: Define Thresholds (75% of max length, 70% of max area)
    # If a grain is smaller than these values, it is considered broken
    length_limit = max_length * 0.75
    area_limit = max_area * 0.70
    
    # STEP 4: Compare each grain against the limits
    for grain in grain_features:
        if grain['length'] < length_limit or grain['area'] < area_limit:
            grain['classification'] = 'Broken'
        else:
            grain['classification'] = 'Full'
            
    return grain_features, (max_length, max_area)
