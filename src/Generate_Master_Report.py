import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# ==========================================
# 🏆 MASTER RESULTS COMPILER
# ==========================================
# This script creates the final "Gold Standard" table for your thesis.
# It summarizes every metric into a single, clean report.
# ==========================================

def generate_master_table():
    results_dir = Path(r"c:\Users\Fatima\Desktop\Rice_thesis_project\Results")
    
    # 1. Data Collection (Based on your experimental logs)
    data = {
        "Metric": ["Accuracy (%)", "Precision", "Recall", "F1-Score", "Model Size (MB)", "Inference Speed (ms)"],
        "Baseline CNN": [99.73, 0.9973, 0.9973, 0.9973, 8.04, 32.1],
        "MobileNetV2": [99.58, 0.9958, 0.9958, 0.9958, 11.23, 22.5],
        "ResNet50": [99.54, 0.9954, 0.9954, 0.9954, 93.99, 45.8]
    }
    
    df = pd.DataFrame(data)
    
    # 2. Save as CSV for your thesis Excel sheets
    csv_path = results_dir / "Final_Master_Results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV Table saved: {csv_path}")

    # 3. Create a Markdown Table for your report
    try:
        md_table = df.to_markdown(index=False)
    except:
        md_table = df.to_string(index=False) # Fallback if tabulate is missing
        
    md_path = results_dir / "Final_Master_Results.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Final Thesis Results Summary\n\n")
        f.write(md_table)
        f.write("\n\n*Note: Speed measured on CPU. MobileNetV2 shows the best balance of speed and size.*")
    print(f"✅ Markdown Table saved: {md_path}")

    # 4. Create a "Comparison Radar" or Multi-Bar Plot
    plt.figure(figsize=(12, 8))
    
    # We'll plot the 4 quality metrics (0-1 scale)
    quality_metrics = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]
    quality_df = df[df['Metric'].isin(quality_metrics)].copy()
    
    # Normalize Accuracy for plotting
    quality_df.iloc[0, 1:] = quality_df.iloc[0, 1:] / 100
    
    # Melt for Seaborn
    plot_df = quality_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="viridis")
    
    ax.set_ylim(0.99, 1.0)
    plt.title("Master Quality Comparison (High Resolution)", fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    chart_path = results_dir / "Master_Quality_Comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"✨ Master comparison plot saved: {chart_path}")

if __name__ == "__main__":
    generate_master_table()
