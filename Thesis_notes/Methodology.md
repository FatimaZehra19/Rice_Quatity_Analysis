# Methodology

The development of the proposed system for rice quality assessment is divided into two primary pipelines: **Rice Variety Classification** using Deep Learning and **Broken Grain Detection** using a Hybrid Computer Vision Approach.

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
- **Hyperparameter Configuration**: Training was conducted using mini-batch gradient descent with a batch size of **32**.
- **Model Checkpointing**: The training process monitored validation accuracy across multiple epochs. The state-dictionary of the model achieving the **highest validation accuracy** was persisted as the "best-performing" candidate to prevent overfitting.

### 1.4 Evaluation Framework
- **Performance Verification**: Post-training, the optimized weights were evaluated on the independent test subset, ensuring that the accuracy metrics represent the model's ability to generalize to novel images.
- **Metrics**: Performance was quantified using standard classification metrics including accuracy, precision, and recall.

## 2. Broken Grain Detection Methodology

A hybrid approach combining computer vision segmentation and geometric feature analysis was used to detect broken rice grains. The methodology is divided into several modular stages for robust and clean image processing:

- **Stage 1: Image Preprocessing**:
    - High-quality grayscale conversion was performed to simplify the image data.
    - Gaussian blurring was applied to reduce noise and artifacts.
    - Otsu's thresholding was used for determining optimal thresholds for image segmentation (grains as foreground).
    - Morphological opening was as a final cleanup step to remove tiny background specks or dust.

- **Stage 2: Watershed-Based Segmentation**:
    - Touching or overlapping grains were separated using the Watershed algorithm. 
    - This involved calculating a distance transform measuring the distance from each grain pixel to the nearest background pixel.
    - "Local peaks" were detected within these distance maps using a minimum distance constraint (min_distance=50) to prevent over-segmentation.
    - These peaks acted as markers for Watershed boundary definition.

- **Stage 3: Geometric Feature Extraction**:
    - Extracted properties: **Area**, **Length** (Major Axis), **Width** (Minor Axis), and **Aspect Ratio**.
    - Centroids were identified for each grain for visual labeling ('F' for Full, 'B' for Broken).

- **Stage 4: Relative Classification**:
    - A dynamic approach was used where grain quality was judged relative to the reference size.
    - A "Full Grain" reference was established by finding the maximum grain size in the image sample.
    - Thresholds (75% of max length / 70% of max area) determined if a grain was classified as "Broken."

- **Stage 5: Batch Reporting and Visualization**:
    - Results were logged in an audit log across all five varieties.
    - Visual bar charts were generated to compare broken grain ratios between categories.

## 3. Explainable AI (XAI) and Interpretability

To ensure the transparency and reliability of the classification models, Explainable AI (XAI) techniques were integrated into the research workflow.

### 3.1 Visual Interpretability with Grad-CAM
- **Objective**: To identify which spatial regions of a rice image the CNN (MobileNetV2 or ResNet50) prioritizes when predicting a specific variety.
- **Gradient-weighted Class Activation Mapping (Grad-CAM)**: This technique utilizes the gradients of the target class (e.g., *Basmati*), flowing into the final convolutional layer to produce a localization map.
- **Target Layers**:
  - **MobileNetV2**: The final expansion/depthwise-convolutional layer in the feature extractor (`features[18][0]`).
  - **ResNet50**: The final bottleneck block in the fourth residual layer (`layer4[-1]`).
- **Heatmap Visualization**: The resulting activation map is superimposed on the original grain image, where warmer colors (red/yellow) indicate higher influence on the model’s classification decision.

### 3.2 Interpretability in Geometric Pipelines
- **Feature-Space Visualization**: For the broken grain detection pipeline, the decision logic is made "explainable" by visualizing the feature space (Length vs. Area).
- **Decision Boundary**: Plotting individual grains against the dynamically calculated "Full Grain" reference thresholds allows for a direct audit of the classification logic.
- **Geometric Annotation**: Automated labeling of each grain with its specific measurements (length/area) provides a clear rationale for every individual "Full" vs. "Broken" classification decision.