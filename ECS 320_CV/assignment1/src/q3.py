
"""
Q3: Apply edge detection filters (Sobel, Laplacian) to all images in genai_image_dataset/dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from scipy.ndimage import sobel, laplace

# Path to dataset
dataset_root = "genai_image_dataset/dataset"

# Output directory
output_dir = "frames/plots/q3"
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

    # Convert to grayscale
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])

    # Sobel edge detection
    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    plt.imshow(sobel_mag, cmap='gray')
    plt.title('Sobel Magnitude')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_sobel.png"))
    plt.close()

    # Laplacian edge detection
    lap = laplace(gray)
    plt.imshow(np.abs(lap), cmap='gray')
    plt.title('Laplacian')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_laplacian.png"))
    plt.close()
