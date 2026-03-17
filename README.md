# Deep Learning Based Rice Variety Classification and Broken Grain Detection

## Project Overview
This project aims to develop a deep learning based system for automatic rice variety classification and broken grain detection using computer vision techniques.

The system uses Convolutional Neural Networks (CNNs) and transfer learning architectures such as ResNet, VGG16, and MobileNet to classify rice varieties from image datasets.

Additionally, computer vision techniques are used to detect broken rice grains for quality assessment.

## Dataset
The project uses the Rice Image Dataset containing five rice varieties:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

Total images: **75,000**

Each class contains **15,000 images** with image size approximately **250 × 250 pixels**.

## Project Structure
Rice_thesis_project
│
├── Dataset
├── Experiments
├── Results
├── src
├── Thesis_notes
└── requirements.txt


## Methodology Pipeline

1. Dataset Exploration
2. Image Preprocessing
3. Watershed Segmentation
4. Deep Learning Model Training (Baseline, MobileNetV2, ResNet50)
5. Explainable AI Analysis (Grad-CAM, Morphology, Geometric Audits)
6. Computational Efficiency Benchmarking
7. Robustness Validation Under Noise/Low-Light

## Key Results Summary

| Model | Accuracy | Size (MB) | Inference Speed |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | 99.73% | 8.04 MB | 32.1 ms |
| **MobileNetV2** | 99.58% | 11.23 MB | 22.5 ms |
| **ResNet50** | 99.54% | 93.99 MB | 45.8 ms |

*Detailed reports available in `/Results` folder.*

## XAI Techniques Implemented

- **Grad-CAM**: Spatial localization of variety features.
- **Morphological Profiling**: Quantifying Slenderness and Area metrics.
- **Decision Auditing**: Explaining human-readable logic for broken grain detection.

## Tools and Libraries

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Author

Fatima Zehra  
MS Artificial Intelligence  
NED University