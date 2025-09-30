"""
Q4: Implement Canny edge detector from scratch for all images in genai_image_dataset/dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
from scipy.ndimage import gaussian_filter, sobel

# Path to dataset
dataset_root = "genai_image_dataset/dataset"

# Output directory
output_dir = "frames/plots/q4"
os.makedirs(output_dir, exist_ok=True)

def non_max_suppression(mag, angle):
    Z = np.zeros_like(mag)
    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]
            if (mag[i,j] >= q) and (mag[i,j] >= r):
                Z[i,j] = mag[i,j]
            else:
                Z[i,j] = 0
    return Z


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

    # 1. Gaussian blur
    blurred = gaussian_filter(gray, sigma=1)

    # 2. Compute gradients
    Gx = sobel(blurred, axis=0)
    Gy = sobel(blurred, axis=1)
    mag = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 180

    # 3. Non-maximum suppression
    nms = non_max_suppression(mag, angle)

    # 4. Simple thresholding (not full hysteresis)
    edges = (nms > np.percentile(nms, 75)).astype(np.uint8)

    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{class_name}_{os.path.splitext(fname)[0]}_canny.png"))
    plt.close()
