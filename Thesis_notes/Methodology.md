## Methodology

- The Rice Image Dataset containing five rice varieties (Arborio, Basmati, Ipsala, Jasmine, and Karacadag) was used for the classification task.

- The dataset was loaded using the PyTorch ImageFolder class, which automatically assigns labels based on the folder structure.

- All images were resized to 224 × 224 pixels to maintain a consistent input size for the convolutional neural network.

- Image pixel values were normalized to improve model convergence and training stability.

- The dataset was divided into training (70%), validation (15%), and testing (15%) subsets to ensure proper model training and unbiased evaluation.

- A baseline convolutional neural network (CNN) was designed for rice variety classification.

- The CNN architecture consists of three convolutional layers followed by max-pooling layers for feature extraction and spatial dimensionality reduction.

- ReLU activation functions were applied after convolutional layers to introduce non-linearity.

- The extracted features were passed through fully connected layers to perform final classification into the five rice varieties.

- The model learns hierarchical visual features such as grain texture, shape, and color patterns from the rice grain images.

- The model was implemented using the PyTorch deep learning framework.

- Training was performed using mini-batch gradient descent with a batch size of 32.

- The Adam optimizer was used to update the model parameters with a learning rate of 0.001.

- CrossEntropyLoss was used as the loss function for the multi-class classification problem.

- The training process was conducted for multiple epochs, during which the training loss and validation accuracy were monitored after each epoch.

- The model achieving the highest validation accuracy during training was saved as the best-performing model.

- After training, the saved model was evaluated on the test dataset, which contains images not seen during the training phase.

- The evaluation process computes classification accuracy on the test set, which represents the final performance of the baseline CNN model.