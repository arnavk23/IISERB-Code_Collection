"""
Q1: Analyze color channels, convert to grayscale, and plot histograms for all images in genai_image_dataset/dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread

# Path to dataset
dataset_root = "genai_image_dataset/dataset"

# Output directory
output_dir = "frames/plots/q1"
os.makedirs(output_dir, exist_ok=True)


# Gather all image paths from all classes
import random
image_paths = []
for class_name in os.listdir(dataset_root):
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
            image_paths.append((class_name, fname, os.path.join(class_dir, fname)))

# Randomly select 200 images
random.seed(42)
selected_images = random.sample(image_paths, min(200, len(image_paths)))

# Process selected images
for class_name, fname, img_path in selected_images:
    try:
        img = imread(img_path)
        if img.ndim == 2:  # grayscale
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:,:,:3]
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        continue

    # Plot color channels
    plt.figure(figsize=(10, 3))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(1, 3, i+1)
        plt.imshow(img[:,:,i], cmap='gray')
        plt.title(f'{color} channel')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_channels.png"))
    plt.close()

    # Convert to grayscale
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_grayscale.png"))
    plt.close()

    # Plot histograms
    plt.figure(figsize=(8, 4))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.hist(img[:,:,i].flatten(), bins=32, alpha=0.5, label=color)
    plt.legend()
    plt.title('Channel Histograms')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_histogram.png"))
    plt.close()
plt.close()

# Plot histograms
plt.figure(figsize=(8, 4))
for i, color in enumerate(['Red', 'Green', 'Blue']):
    plt.hist(img[:,:,i].flatten(), bins=32, alpha=0.5, label=color)
plt.legend()
plt.savefig(os.path.join(output_dir, 'q1_histogram.png'))
plt.close()
plt.title('Color Channel Histograms')
plt.savefig('frames/plots/q1_histogram.png')
plt.close()
