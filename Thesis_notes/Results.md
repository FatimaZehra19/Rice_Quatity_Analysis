# Experimental Results and Discussion

This document summarizes the performance of the implemented deep learning models for rice variety classification.

## 1. Baseline CNN Performance

The Baseline CNN is a custom 4-layer convolutional neural network architecture with batch normalization and dropout for regularization.

### Dataset Split
| Dataset | Number of Images |
|--------|------------------|
| Training Set | 52,500 |
| Validation Set | 11,250 |
| Test Set | 11,250 |

### Test Evaluation Results (Baseline)
- **Test Accuracy:** 99.69%
- **Correct Predictions:** 11,215 / 11,250
- **Precision/Recall:** Balanced performance across all 5 classes (>99.5%).

The baseline model demonstrates exceptional feature extraction capabilities for this dataset, achieving near-perfect accuracy with a relatively lightweight architecture.

---

## 2. ResNet50 Transfer Learning Performance

We utilized a pretrained **ResNet50** architecture, fine-tuned on the rice dataset by replacing the final classification head with a custom fully connected layer (512 units, ReLU, Dropout).

### Test Evaluation Results (ResNet50)
- **Test Accuracy:** 99.55%
- **Correct Predictions:** 11,199 / 11,250
- **Top Performing Class:** Ipsala (1.00 Precision, 99.96% Recall)

### Detailed Metrics
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Arborio | 0.9935 | 0.9917 | 0.9926 |
| Basmati | 0.9954 | 0.9963 | 0.9959 |
| Ipsala | 1.0000 | 0.9996 | 0.9998 |
| Jasmine | 0.9905 | 0.9948 | 0.9926 |
| Karacadag| 0.9978 | 0.9947 | 0.9963 |

---

## 3. Comparative Analysis

### Training Dynamics
As seen in the training curves, the **Baseline CNN** converged faster due to its simpler architecture, while **ResNet50** showed more stable loss reduction over 30 epochs owing to its deep residual connections.

### Model Comparison
- **Accuracy:** Interestingly, the Baseline CNN slightly outperformed ResNet50 on this specific dataset (99.69% vs 99.55%). This suggests that for high-resolution, high-contrast rice grain images, a simpler architecture may be sufficient and less prone to overfitting than a deep 50-layer network.
- **Robustness:** ResNet50 showed superior performance on the "Ipsala" variety, achieving perfect precision, indicating its strength in identifying distinctive class features.
- **Micro-Metrics:** As shown in the "Model Performance Comparison" graph, the Baseline CNN maintains a slight edge in Precision, Recall, and F1-score across the entire test set.


### Visualizations
- **Model Performance Metrics:** The bar chart below compares the overall Precision, Recall, and F1-score for both models, highlighting the high consistency of the Baseline CNN.
- **Confusion Matrices:** Both models show minimal misclassification, with most errors occurring between "Arborio" and "Jasmine" varieties due to their similar visual textures.
- **Training Curves:** Validation accuracy for both models plateaued after ~15 epochs, indicating that the learning rate scheduler effectively optimized convergence.

![Model Performance Comparison](../Results/model_performance_comparison.png)


## 4. Conclusion
Both models are highly suitable for automated rice variety classification. The **Baseline CNN** offers a more efficient (faster inference) solution, while **ResNet50** provides a robust transfer-learning alternative that could potentially generalize better to broader, unseen datasets.