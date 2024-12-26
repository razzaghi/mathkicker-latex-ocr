import os
import shutil
import random

# Set the paths for your images directory and the target directories for train, test, and validation
images_dir = '/home/ubuntu/latex/scaled_images'  # Update with the correct path to your images
output_dir = '/home/ubuntu/latex/mathkicker_dataset'  # Update with the desired output directory path

# Directory names for train, test, and validation
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
val_dir = os.path.join(output_dir, 'val')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of all image files in the images directory
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

# Set the splitting ratios (you can modify these)
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Shuffle the list of image files
random.shuffle(image_files)

# Calculate the number of images for each set
num_images = len(image_files)
train_size = int(num_images * train_ratio)
test_size = int(num_images * test_ratio)
val_size = num_images - train_size - test_size

# Split the image files into train, test, and val sets
train_files = image_files[:train_size]
test_files = image_files[train_size:train_size + test_size]
val_files = image_files[train_size + test_size:]

# Function to copy files to a target directory
def copy_files(file_list, target_dir):
    for file in file_list:
        shutil.copy(os.path.join(images_dir, file), os.path.join(target_dir, file))

# Copy the images to their respective directories
copy_files(train_files, train_dir)
copy_files(test_files, test_dir)
copy_files(val_files, val_dir)

print(f"Images have been split into train, test, and val directories.")
