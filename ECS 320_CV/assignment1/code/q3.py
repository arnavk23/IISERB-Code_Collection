"""
Q3: Apply filters (3x3, 5x5, 7x7) and perform edge detection (Sobel, Laplacian) on clean and noisy images.

This script demonstrates:
1. Loading clean and noisy grayscale images.
2. Applying mean filters of different sizes (3x3, 5x5, 7x7) to smooth images.
3. Performing edge detection using Sobel (horizontal and vertical) and Laplacian operators, all implemented from scratch.
4. Visualizing and saving the results for each filter and edge detection method.
5. Saving all plots and processed images in organized folders for reporting.

All image processing is done from scratch, without OpenCV or PIL.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(img, ksize):
    """
    Apply a mean filter (box filter) of size ksize x ksize to the input image.
    Args:
        img: 2D numpy array (grayscale image)
        ksize: int, size of the filter kernel (must be odd)
    Returns:
        filtered: 2D numpy array, filtered image
    """
    pad = ksize // 2
    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    filtered = np.zeros_like(img)
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            filtered[i, j] = np.mean(img_padded[i:i+ksize, j:j+ksize])
    return filtered.astype(np.uint8)

def sobel_edge(img, direction='x'):
    """
    Apply Sobel edge detection in the specified direction.
    Args:
        img: 2D numpy array (grayscale image)
        direction: 'x' for horizontal, 'y' for vertical
    Returns:
        edge: 2D numpy array, edge map
    """
    if direction == 'x':
        kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    else:
        kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    pad = 1
    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    edge = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            edge[i, j] = np.sum(img_padded[i:i+3, j:j+3] * kernel)
    return np.clip(np.abs(edge), 0, 255).astype(np.uint8)

def laplacian_edge(img):
    """
    Apply Laplacian edge detection to the input image.
    Args:
        img: 2D numpy array (grayscale image)
    Returns:
        edge: 2D numpy array, edge map
    """
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    pad = 1
    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    edge = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            edge[i, j] = np.sum(img_padded[i:i+3, j:j+3] * kernel)
    return np.clip(np.abs(edge), 0, 255).astype(np.uint8)


# Step 1: Load clean and noisy grayscale images using OpenCV
img_bgr = cv2.imread('images/your_photo.jpg')
if img_bgr is None:
    raise FileNotFoundError('Image not found. Please place your photo in the images folder and name it your_photo.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
noisy_img = cv2.imread('images/produced/q2/noisy_grayscale.jpg', cv2.IMREAD_GRAYSCALE)
if noisy_img is None:
    raise FileNotFoundError('Noisy grayscale image not found. Please run Q2 to generate it.')

# Step 2: Prepare images and filter sizes
images = {'Clean': img, 'Noisy': noisy_img}
ksizes = [3, 5, 7]

# Step 3: Apply filters and edge detectors, visualize and save results
for name, image in images.items():
    fig = plt.figure(figsize=(18,12))
    for idx, k in enumerate(ksizes):
        # Apply mean filter
        filtered = apply_filter(image, k)
        # Edge detection
        sobel_x = sobel_edge(filtered, 'x')
        sobel_y = sobel_edge(filtered, 'y')
        laplacian = laplacian_edge(filtered)
        # Plot results
        plt.subplot(len(ksizes), 4, idx*4+1)
        plt.imshow(filtered, cmap='gray')
        plt.title(f'{name} {k}x{k} Filtered')
        plt.axis('off')
        plt.subplot(len(ksizes), 4, idx*4+2)
        plt.imshow(sobel_x, cmap='gray')
        plt.title('Sobel X')
        plt.axis('off')
        plt.subplot(len(ksizes), 4, idx*4+3)
        plt.imshow(sobel_y, cmap='gray')
        plt.title('Sobel Y')
        plt.axis('off')
        plt.subplot(len(ksizes), 4, idx*4+4)
        plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian')
        plt.axis('off')
    plt.suptitle(f'Edge Detection on {name} Image')
    plt.tight_layout()
    fig.savefig(f'images/plots/q3_{name.lower()}_edges.png')  # Save plot for report
    plt.show()


# Step 4: Save sample processed images for report in produced/q3 using OpenCV
cv2.imwrite('images/produced/q3/clean_3x3_filtered.jpg', apply_filter(img, 3))
cv2.imwrite('images/produced/q3/noisy_3x3_filtered.jpg', apply_filter(noisy_img, 3))
cv2.imwrite('images/produced/q3/clean_sobelx.jpg', sobel_edge(img, 'x'))
cv2.imwrite('images/produced/q3/noisy_sobelx.jpg', sobel_edge(noisy_img, 'x'))