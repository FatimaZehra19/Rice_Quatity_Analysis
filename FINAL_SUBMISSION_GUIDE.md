# 🎓 Rice Quality Analysis: Final Submission Guide

This project is a complete AI-driven solution for **Rice Variety Classification** and **Quality Assessment** (Broken Grain Detection), featuring **Explainable AI (XAI)**.

---

## 📂 Project Organization (Cleaned & Optimized)

### 1. 🧪 Experiments & Models (`/Experiments`)
*   Contains the trained `.pth` weight files for all 3 models:
    *   `rice_cnn_baseline_best.pth` (The primary auditor)
    *   `rice_mobilenetv2_transfer_best.pth` (Fast & Efficient)
    *   `rice_resnet50_transfer_best.pth` (High Depth)

### 2. 📊 Final Results & Evidence (`/Results`)
*   **`Efficiency_Comparison.png`**: Shows Model Size vs. Inference Speed.
*   **`Master_Quality_Comparison.png`**: The final performance chart (Precision/Recall/F1).
*   **`Final_Master_Results.md`**: A clean table summarizing every experiment.
*   **`/XAI_Reports`**: The heatmaps (Grad-CAM) and morphology charts.
*   **`/Final_Broken_Grain_Report`**: Quality audits for each rice variety.

### 3. 🐍 Core Scripts (`/src` and `/Broken_Grains_Analysis`)
*   **`Main_Analysis.py`**: The master pipeline (Preprocessing -> Segmentation -> Classification -> XAI).
*   **`Compare_XAI_Models.py`**: Comparison of how different AI "see" the grains.
*   **`Variety_Morphology_XAI.py`**: Scientific proof of physical differences (Slenderness/Size).
*   **`Compare_Efficiency.py`**: Benchmarks speed and model footprint.

---

## 🚀 How to Run "The Big Demo"

To generate the complete results for your thesis:

1.  **Run the Performance Benchmark**:
    ```bash
    python src/Compare_Efficiency.py
    ```
2.  **Generate the Master Table**:
    ```bash
    python src/Generate_Master_Report.py
    ```
3.  **Run the Quality Audit (XAI)**:
    ```bash
    python Broken_Grains_Analysis/Main_Analysis.py
    ```

---

## 💡 Key Findings for your Presentation
*   **Inference Precision**: All models achieved >99% accuracy on the target dataset.
*   **Localization**: The **Baseline CNN** provided the most precise spatial localization in Grad-CAM heatmaps.
*   **Efficiency**: **MobileNetV2** is the best candidate for real-time mobile app deployment as it is ~50% faster than ResNet50.
*   **Quality**: Basmati showed the highest "Broken" percentage in samples (~2.0%), while Ipsala showed the most consistent full-grain quality.

---
**Author:** Fatima Zehra  
**Degree:** MS Artificial Intelligence  
**Institution:** NED University
