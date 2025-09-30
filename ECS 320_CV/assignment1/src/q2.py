
"""
Q2: Add Gaussian noise to all images in genai_image_dataset/dataset and plot noisy channels and histograms.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread

# Path to dataset
dataset_root = "genai_image_dataset/dataset"

# Output directory
output_dir = "frames/plots/q2"
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

    # Add Gaussian noise
    noise = np.random.normal(0, 25, img.shape)
    noisy_img = np.clip(img + noise, 0, 1 if img.dtype == np.float32 else 255)
    if img.dtype != np.float32:
        noisy_img = noisy_img.astype(np.uint8)

    # Plot noisy color channels
    plt.figure(figsize=(10, 3))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(1, 3, i+1)
        plt.imshow(noisy_img[:,:,i], cmap='gray')
        plt.title(f'Noisy {color}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_noisy_channels.png"))
    plt.close()

    # Plot noisy histograms
    plt.figure(figsize=(8, 4))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.hist(noisy_img[:,:,i].flatten(), bins=32, alpha=0.5, label=color)
    plt.legend()
    plt.title('Noisy Channel Histograms')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_noisy_histogram.png"))
    plt.close()
