import os 
import torch
from torchvision import datasets as tv_datasets ,transforms
from torch.utils.data import DataLoader ,random_split

def get_data_loaders(batch_size=32, num_workers=None):
    """
    Loads the rice image dataset using ImageFolder, applies preprocessing,
    splits into train/validation/test sets, and returns DataLoaders.

    Dataset structure:
        Dataset/Rice_Image_Dataset/
        ├── Arborio/
        ├── Basmati/
        ├── Ipsala/
        ├── Jasmine/
        └── Karacadag/
    """

    # Get Project Root Directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up to project root

    # Dataset path - CORRECTED for 75,000 images structure
    dataset_path = os.path.join(project_root, "Dataset", "Rice_Image_Dataset")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}\n"
                              f"Expected structure:\n"
                              f"  Dataset/Rice_Image_Dataset/Arborio/\n"
                              f"  Dataset/Rice_Image_Dataset/Basmati/\n"
                              f"  Dataset/Rice_Image_Dataset/Ipsala/\n"
                              f"  Dataset/Rice_Image_Dataset/Jasmine/\n"
                              f"  Dataset/Rice_Image_Dataset/Karacadag/")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
                                    ])

    # Load the dataset
    dataset = tv_datasets.ImageFolder(root=dataset_path, transform=transform)

    # Size of the dataset
    dataset_size = len(dataset)


    # Split the dataset
    train_size = int(0.7 * dataset_size)  # 70% for training
    val_size = int(0.15 * dataset_size)   # 15% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 15% for testing

    generator = torch.Generator().manual_seed(42)  # For reproducibility

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders for each split (optimized for large datasets)
    # num_workers=4 enables parallel data loading for faster training
    # pin_memory=True if GPU available for faster GPU memory transfer
    pin_mem = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 4 if os.cpu_count() >= 8 else 2  # Adaptive based on CPU cores

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_mem)

    # Print dataset info
    print(f"[Dataset Loaded] {dataset_size:,} total images  |  Classes: {dataset.classes}")
    print(f"[Split] Train: {train_size:,}  |  Val: {val_size:,}  |  Test: {test_size:,}")
    print(f"[DataLoaders] Workers: {num_workers}  |  Pin Memory: {pin_mem}")

    return train_loader, val_loader, test_loader, dataset.classes
    
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_data_loaders(batch_size=32)
    # Printing Dataset Information
    print("Total number of classes in the dataset:", len(class_names))
    print("Number of images in the training set:", len(train_loader.dataset))
    print("Number of images in the validation set:", len(val_loader.dataset))
    print("Number of images in the test set:", len(test_loader.dataset))