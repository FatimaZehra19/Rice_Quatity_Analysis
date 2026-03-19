import numpy as np

# Predefined Variety Characteristics for Better Quality Analysis
# Values are based on typical proportions (L/W ratio) for these classes
VARIETY_DATA = {
    'Arborio': {'min_aspect_ratio': 1.6, 'expected_area_ratio': 0.70},
    'Basmati': {'min_aspect_ratio': 3.0, 'expected_area_ratio': 0.65},
    'Ipsala': {'min_aspect_ratio': 2.5, 'expected_area_ratio': 0.75},
    'Jasmine': {'min_aspect_ratio': 2.8, 'expected_area_ratio': 0.70},
    'Karacadag': {'min_aspect_ratio': 1.5, 'expected_area_ratio': 0.60},
}

def classify_grains(grain_features, variety_name=None):
    """
    Decides if a grain is 'Full' or 'Broken'.
    Uses variety-specific reference data for professional-grade results.
    """
    if len(grain_features) == 0:
        return [], (0, 0)
    
    # STEP 1: Find sample reference (still useful as a secondary check)
    max_length = max(g['length'] for g in grain_features)
    max_area = max(g['area'] for g in grain_features)
    
    # STEP 2: Use variety-aware logic
    variety_info = VARIETY_DATA.get(variety_name, {'expected_area_ratio': 0.70})
    
    # Thresholds: If specific variety is known, we can be more strict
    length_limit = max_length * 0.75
    area_limit = max_area * variety_info['expected_area_ratio']
    
    # STEP 3: Classification pass
    for grain in grain_features:
        # A grain is broken if it's much smaller than the largest in the sample
        # OR if its aspect ratio is way off from the variety standard (indicating broken pieces)
        is_small = grain['area'] < area_limit or grain['length'] < length_limit
        
        # If we know the variety, we can also check for shape integrity (Aspect Ratio)
        # Note: we use a relaxed check here as broken grains often have low aspect ratios
        is_deformed = False
        if variety_name in VARIETY_DATA:
            # If it's supposed to be long (Basmati) but is round, it's likely a broken piece
            target_ar = VARIETY_DATA[variety_name]['min_aspect_ratio']
            if grain['aspect_ratio'] < target_ar * 0.6: # 40% deviation allowed
                is_deformed = True

        if is_small:
            grain['classification'] = 'Broken'
        elif variety_name in VARIETY_DATA:
            # High tolerance for shape (50% deviation)
            target_ar = VARIETY_DATA[variety_name]['min_aspect_ratio']
            if grain['aspect_ratio'] < target_ar * 0.5: 
                grain['classification'] = 'Broken'
            else:
                grain['classification'] = 'Full'
        else:
            grain['classification'] = 'Full'
            
    return grain_features, (max_length, max_area)
