import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader , random_split

# Path to the dataset
# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go to project root
project_root = os.path.dirname(current_dir)

# Dataset path
dataset_path = os.path.join(project_root, "Dataset", "Rice_Image_Dataset")

print("Dataset path:", dataset_path)

# Image transformations 
transfrom = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
                                ])

# Load the dataset 
datasets = datasets.ImageFolder(root=dataset_path, transform=transfrom)

# Size of the dataset
dataset_size = len(datasets)


# Split the dataset 
train_size = int(0.7 * dataset_size)  # 70% for training
val_size = int(0.15 * dataset_size)   # 15% for validation
test_size = dataset_size - train_size - val_size  # Remaining 15% for testing

train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])

# Create DataLoaders for each split
# num_workers=4: parallel data loading (macOS: use 'fork' to avoid spawn crash)
# pin_memory=False: MPS does not support pinned memory (only CUDA does)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=False,
                          persistent_workers=True, multiprocessing_context='fork')
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=False,
                          persistent_workers=True, multiprocessing_context='fork')
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=False,
                          persistent_workers=True, multiprocessing_context='fork')

# Printing Dataset Information
print("Total number of images in the dataset:", dataset_size)
print("Number of images in the training set:", train_size)
print("Number of images in the validation set:", val_size)
print("Number of images in the test set:", test_size)