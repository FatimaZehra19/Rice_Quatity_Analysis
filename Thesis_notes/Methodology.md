## Methodology 

- Dataset was loaded using PyTorch ImageFolder class. 
- Images were resized to 224×224 pixels and normalized. 
- The dataset was split into training (70%), validation (15%), and testing (15%) sets.
- A baseline convolutional neural network (CNN) was designed for rice variety classification. 
- The architecture consists of three convolutional layers followed by max-pooling operations and fully connected layers. 
- The model learns hierarchical visual features from rice grain images.
- A baseline convolutional neural network was trained using the PyTorch framework. 
- Training was performed using mini-batch gradient descent with a batch size of 32. 
- The model was optimized using the Adam optimizer with a learning rate of 0.001. 
- CrossEntropyLoss was used as the loss function for multi-class classification. 

