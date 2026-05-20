# Methodology

The development of the proposed system focuses on **Rice Variety Classification** using Deep Learning, with plans to extend to grain quality assessment in future work.

## 1. Rice Variety Classification Methodology

This section outlines the workflow for developing the automated classification system for different rice varieties.

### 1.1 Data Acquisition and Preprocessing
- **Source Material**: The Rice Image Dataset, comprising five distinct varieties—Arborio, Basmati, Ipsala, Jasmine, and Karacadag—was utilized for the classification task.
- **Data Loading**: The dataset was indexed using the PyTorch `ImageFolder` class, which facilitated automated label mapping based on the hierarchical folder structure.
- **Normalization and Resizing**: All images underwent standard resizing to **224 × 224 pixels** to ensure a uniform input dimension for the convolutional layers. Pixel intensity values were normalized to accelerate model convergence and maintain training stability.
- **Dataset Partitioning**: To ensure an unbiased evaluation, the data was stratified into three subsets: **Training (70%)**, **Validation (15%)**, and **Testing (15%)**.

### 1.2 Network Architecture and Feature Learning
- **Baseline Design**: A custom Convolutional Neural Network (CNN) architecture was designed as the project baseline.
- **Layer Stacking**: The architecture comprises triplets of convolutional layers followed by Max-Pooling operations for effective spatial dimensionality reduction and robust feature extraction.
- **Non-Linearity**: ReLU (Rectified Linear Unit) activation functions were integrated to model non-linear relationships within the visual data.
- **Classification Head**: Extracted global features were mapped to a final fully connected layer for probabilistic classification across the five rice categories. 
- **Learning Objective**: The model was tasked with identifying hierarchical visual features, including grain texture patterns, aspect ratios, and distinctive color distributions unique to each variety.

### 1.3 Model Training Strategy and Optimization
- **Optimization Algorithm**: The **Adam Optimizer** was employed for parameter updates with a learning rate of **0.001**.
- **Loss Function**: `CrossEntropyLoss` was used to measure the discrepancy between the predicted and actual variety labels.
- **Hyperparameter Configuration**: Training was conducted using mini-batch gradient descent with:
  - **Batch Size**: 64 (increased from 32 to leverage the 75,000-image dataset for better GPU utilization)
  - **Number of Epochs**: 25 (reduced from 30; larger datasets converge faster)
  - **Learning Rate Scheduler**: StepLR with decay factor 0.5 every 10 epochs
  - **Weight Decay**: 1e-4 (L2 regularization)

- **Data Augmentation**: To improve model robustness and prevent overfitting on the large dataset, the following augmentation techniques were applied exclusively to the training set:
  - **RandomHorizontalFlip** (p=0.5): Handles image orientation variations
  - **RandomRotation** (±10°): Provides robustness to grain rotation
  - **ColorJitter** (brightness, contrast, saturation ±0.2): Simulates lighting variations
  - **GaussianBlur** (kernel=3, σ=0.1-0.2): Reduces overfitting to image noise
  - Validation and test sets used **standard transforms only** (no augmentation) for fair evaluation

- **Model Checkpointing**: The training process monitored validation accuracy across multiple epochs. The state-dictionary of the model achieving the **highest validation accuracy** was persisted as the "best-performing" candidate to prevent overfitting.

- **Transfer Learning Strategy**: For MobileNetV2 and ResNet50, all convolutional backbone layers were **frozen** and only the custom classification head was trained (head-only fine-tuning). This is a conservative transfer learning strategy that preserves ImageNet-derived feature representations. The trade-off is that the frozen layers were never adapted to the rice domain, which can limit performance on datasets with narrow visual characteristics. Full end-to-end fine-tuning — where all layers are unfrozen and trained jointly with a reduced learning rate — is expected to yield higher accuracy for the transfer learning models and is recommended as a direction for future work.

### 1.4 Evaluation Framework
- **Performance Verification**: Post-training, the optimized weights were evaluated on the independent test subset, ensuring that the accuracy metrics represent the model's ability to generalize to novel images.
- **Metrics**: Performance was quantified using standard classification metrics including accuracy, precision, and recall.

## 2. Explainable AI (XAI) and Interpretability

To ensure the transparency and reliability of the classification models, Explainable AI (XAI) techniques were integrated into the research workflow.

### 2.1 Visual Interpretability with Grad-CAM
- **Objective**: To identify which spatial regions of a rice image the CNN (MobileNetV2 or ResNet50) prioritizes when predicting a specific variety.
- **Gradient-weighted Class Activation Mapping (Grad-CAM)**: This technique utilizes the gradients of the target class (e.g., *Basmati*), flowing into the final convolutional layer to produce a localization map.
- **Target Layers**:
  - **MobileNetV2**: The final expansion/depthwise-convolutional layer in the feature extractor (`features[18][0]`).
  - **ResNet50**: The final bottleneck block in the fourth residual layer (`layer4[-1]`).
- **Heatmap Visualization**: The resulting activation map is superimposed on the original grain image, where warmer colors (red/yellow) indicate higher influence on the model’s classification decision.
