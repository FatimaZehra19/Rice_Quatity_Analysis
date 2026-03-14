import torch    
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RiceCNN(nn.Module):
    """Improved custom CNN with batch normalization and dropout for better regularization"""
    def __init__(self, num_classes=5):
        super(RiceCNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.25)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.25)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256, 512)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(self.dropout1(self.bn1(F.relu(self.conv1(x)))))
        
        # Block 2
        x = self.pool(self.dropout2(self.bn2(F.relu(self.conv2(x)))))
        
        # Block 3
        x = self.pool(self.dropout3(self.bn3(F.relu(self.conv3(x)))))
        
        # Block 4
        x = self.pool(self.dropout4(self.bn4(F.relu(self.conv4(x)))))
        
        # Global Average Pooling (works with any input size)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc_dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":

    model = RiceCNN()
    print(model)
    