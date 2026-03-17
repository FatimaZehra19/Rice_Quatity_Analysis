import cv2
import os
import glob
import json
import matplotlib.pyplot as plt
from Preprocessing import preprocess_image
from Segmentation import segment_grains
from Feature_Analysis import extract_features
from Classification import classify_grains

def analyze_rice_sample(image_path, save_image=False, output_folder=None):
    """
    Runs the complete analysis on a single rice image.
    """
    image_name = os.path.basename(image_path)
    
    # 1. Clean the image
    binary, original = preprocess_image(image_path)
    if binary is None: return None
    
    # 2. Separate touching grains
    labels, _ = segment_grains(binary)
    
    # 3. Measure grain sizes
    features = extract_features(labels)
    
    # 4. Classify as Full or Broken
    classified_data, (max_len, max_area) = classify_grains(features)
    
    # 5. Count Results
    full_count = 0
    broken_count = 0
    for grain in classified_data:
        if grain['classification'] == 'Full':
            full_count += 1
        else:
            broken_count += 1
            
    total = full_count + broken_count
    percentage_broken = (broken_count / total * 100) if total > 0 else 0
    
    # 6. Optional: Save only ONE result image per category to save memory
    if save_image and output_folder:
        result_img = original.copy()
        for grain in classified_data:
            color = (0, 255, 0) if grain['classification'] == 'Full' else (0, 0, 255)
            cv2.circle(result_img, grain['centroid'], 3, color, -1)
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, f"sample_{image_name}"), result_img)
    
    return {
        "Image": image_name,
        "Total Grains": total,
        "Full Grains": full_count,
        "Broken Grains": broken_count,
        "% Broken": round(percentage_broken, 2)
    }

def main():
    # SETTINGS
    dataset_root = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Dataset\Rice_Image_Dataset"
    output_path = r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results\Final_Broken_Grain_Report"
    os.makedirs(output_path, exist_ok=True)
    
    categories = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    summary_data = {}

    print(f"🚀 Starting Full Dataset Analysis...")
    
    for category in categories:
        print(f"\n📂 Processing Category: {category}")
        cat_path = os.path.join(dataset_root, category)
        images = glob.glob(os.path.join(cat_path, "*.jpg"))[:50] # Process 50 images for a good sample
        
        cat_results = []
        for i, img_path in enumerate(images):
            # Save only the FIRST image of each category to save memory
            save_this_img = (i == 0) 
            data = analyze_rice_sample(img_path, save_image=save_this_img, output_folder=output_path)
            if data:
                cat_results.append(data)
        
        # Calculate Average for this Category
        avg_broken = sum(d['% Broken'] for d in cat_results) / len(cat_results) if cat_results else 0
        summary_data[category] = {
            "avg_broken_percent": round(avg_broken, 2),
            "total_images": len(cat_results),
            "details": cat_results
        }
        print(f"📊 {category} Average: {round(avg_broken, 2)}% broken")

    # 1. Save results to JSON (saves memory, stores logs)
    json_path = os.path.join(output_path, "analysis_logs.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    # 2. Create and Save the Bar Chart
    names = list(summary_data.keys())
    values = [info["avg_broken_percent"] for info in summary_data.values()]
    
    plt.figure(figsize=(10, 6))
    bar_colors = ['#4CAF50', '#2E7D32', '#8BC34A', '#CDDC39', '#1B5E20']
    plt.bar(names, values, color=bar_colors)
    plt.xlabel('Rice Variety')
    plt.ylabel('Average Broken Grains (%)')
    plt.title('Broken Grain Analysis Across Rice Varieties')
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.2, f"{v}%", ha='center', fontweight='bold')

    chart_path = os.path.join(output_path, "broken_grain_comparison.png")
    plt.savefig(chart_path)
    
    print(f"\n✨ Analysis Completed!")
    print(f"📁 JSON Log saved: {json_path}")
    print(f"📈 Chart Saved: {chart_path}")

if __name__ == "__main__":
    main()
