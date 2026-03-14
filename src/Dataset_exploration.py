import os
import cv2
import matplotlib.pyplot as plt

# Path to the dataset
# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go to project root
project_root = os.path.dirname(current_dir)

# Dataset path
dataset_path = os.path.join(project_root, "Dataset", "Rice_Image_Dataset")

print("Dataset path:", dataset_path)

classes = os.listdir(dataset_path)
print("Classes in the dataset:", classes)
print("Number of classes:", len(classes))
print()

# Count the number of images in each class
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    num_images = len(os.listdir(class_path))
    print(f"Number of images in class '{cls}': {num_images}")

print("\nDataset exploration completed. \n ")

# Display a few sample images
plt.figure(figsize=(10, 6))
for i, cls in enumerate(classes):
    class_path = os.path.join(dataset_path, cls)
    sample_image = os.listdir(class_path)[0]  # Get the first image from the class
    img_path = os.path.join(class_path, sample_image)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying
    plt.subplot(1, len(classes), i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Check the dimensions of a sample image
sample_classes = classes[0]  # Choose the first class for checking
sample_image = os.listdir(os.path.join(dataset_path, sample_classes))[0]  # Get the first image from the class
sample_path = os.path.join(dataset_path, sample_classes, sample_image)
img = cv2.imread(sample_path)

print("Sample Image Shape(Height, Width, Channels):", img.shape)
