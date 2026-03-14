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


class RiceResNet50(nn.Module):
    """Transfer Learning with ResNet50 - Better for your large dataset"""
    def __init__(self, num_classes=5, pretrained=True):
        super(RiceResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer for 5 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)


class RiceVGG16(nn.Module):
    """Transfer Learning with VGG16 - Good alternative"""
    def __init__(self, num_classes=5, pretrained=True):
        super(RiceVGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        
        # Modify classifier
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)


class RiceMobileNet(nn.Module):
    """Transfer Learning with MobileNet - Lightweight, faster inference"""
    def __init__(self, num_classes=5, pretrained=True):
        super(RiceMobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

    
if __name__ == "__main__":
    print("=== Model Architectures ===\n")
    
    print("1. Improved Custom CNN:")
    model = RiceCNN()
    print(model)
    
    print("\n2. ResNet50 (Transfer Learning):")
    model_resnet = RiceResNet50()
    print(f"ResNet50 loaded. Total parameters: {sum(p.numel() for p in model_resnet.parameters()):,}")
    
    print("\n3. VGG16 (Transfer Learning):")
    model_vgg = RiceVGG16()
    print(f"VGG16 loaded. Total parameters: {sum(p.numel() for p in model_vgg.parameters()):,}")
    
    print("\n4. MobileNet (Transfer Learning - Lightweight):")
    model_mobile = RiceMobileNet()
    print(f"MobileNet loaded. Total parameters: {sum(p.numel() for p in model_mobile.parameters()):,}")