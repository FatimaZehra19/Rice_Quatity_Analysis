## Baseline CNN Performance

- Dataset path: `Dataset/Rice_Image_Dataset`
- Total number of images in the dataset: **75,000**

### Dataset Split

| Dataset | Number of Images |
|--------|------------------|
| Training Set | 52,500 |
| Validation Set | 11,250 |
| Test Set | 11,250 |

### Test Evaluation Results

- Total Test Samples: **11,250**
- Correct Predictions: **11,215**
- Test Accuracy: **99.69%**

The trained baseline CNN model demonstrates strong classification performance for rice grain varieties, correctly classifying **11,215 out of 11,250 test images**, resulting in a **test accuracy of 99.69%**.

### Confusion Matrix

A confusion matrix was generated to analyze the classification performance of the baseline CNN model across all rice varieties.  
The matrix shows the number of correct and incorrect predictions for each class, allowing detailed inspection of model errors.

The confusion matrix indicates that the model correctly classifies almost all rice varieties, with very few misclassifications, consistent with the high test accuracy of **99.69%**.