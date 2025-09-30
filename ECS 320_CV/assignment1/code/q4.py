"""
Q4: Implement Canny edge detector from scratch and apply to clean and noisy images.

This script demonstrates:
1. Loading clean and noisy grayscale images.
2. Implementing the Canny edge detector step-by-step:
    - Gaussian smoothing
    - Gradient magnitude and orientation (Sobel)
    - Non-maximum suppression
    - Hysteresis thresholding
3. Visualizing and saving the results for clean and noisy images.
4. Saving all plots and processed images in organized folders for reporting.

All image processing is done from scratch, without OpenCV or PIL.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    """
    Generate a 2D Gaussian kernel for smoothing.
    Args:
        size: int, size of the kernel (must be odd)
        sigma: float, standard deviation of the Gaussian
    Returns:
        kernel: 2D numpy array
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve(img, kernel):
    """
    Apply a 2D convolution between an image and a kernel.
    Args:
        img: 2D numpy array (grayscale image)
        kernel: 2D numpy array
    Returns:
        result: 2D numpy array, filtered image
    """
    pad = kernel.shape[0] // 2
    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(img_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return result

def sobel_gradients(img):
    """
    Compute gradient magnitude and orientation using Sobel filters.
    Args:
        img: 2D numpy array (grayscale image)
    Returns:
        mag: 2D numpy array, gradient magnitude
        theta: 2D numpy array, gradient orientation (radians)
    """
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    mag = np.hypot(Gx, Gy)
    mag = mag / mag.max() * 255
    theta = np.arctan2(Gy, Gx)
    return mag, theta

def non_max_suppression(mag, theta):
    """
    Apply non-maximum suppression to thin edges.
    Args:
        mag: 2D numpy array, gradient magnitude
        theta: 2D numpy array, gradient orientation (radians)
    Returns:
        Z: 2D numpy array, thinned edge map
    """
    Z = np.zeros_like(mag, dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            q = 255
            r = 255
            # Determine neighbors based on edge direction
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]
            if (mag[i,j] >= q) and (mag[i,j] >= r):
                Z[i,j] = mag[i,j]
            else:
                Z[i,j] = 0
    return Z

def hysteresis(img, low, high):
    """
    Apply hysteresis thresholding to finalize edge map.
    Args:
        img: 2D numpy array, thinned edge map
        low: int, low threshold
        high: int, high threshold
    Returns:
        res: 2D numpy array, final edge map
    """
    strong = 255
    weak = 75
    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    # Track edges by hysteresis: connect weak edges to strong ones
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if res[i,j] == weak:
                if ((res[i+1, j-1] == strong) or (res[i+1, j] == strong) or (res[i+1, j+1] == strong)
                    or (res[i, j-1] == strong) or (res[i, j+1] == strong)
                    or (res[i-1, j-1] == strong) or (res[i-1, j] == strong) or (res[i-1, j+1] == strong)):
                    res[i,j] = strong
                else:
                    res[i,j] = 0
    return res

def canny_edge(img, low=50, high=100):
    """
    Full Canny edge detection pipeline.
    Args:
        img: 2D numpy array (grayscale image)
        low: int, low threshold for hysteresis
        high: int, high threshold for hysteresis
    Returns:
        edges: 2D numpy array, final edge map
    """
    # 1. Gaussian smoothing
    blur = convolve(img, gaussian_kernel(5, 1))
    # 2. Gradient magnitude and orientation
    mag, theta = sobel_gradients(blur)
    # 3. Non-maximum suppression
    nms = non_max_suppression(mag, theta)
    # 4. Hysteresis thresholding
    edges = hysteresis(nms, low, high)
    return edges


# Step 1: Load clean and noisy grayscale images using OpenCV
img_bgr = cv2.imread('images/your_photo.jpg')
if img_bgr is None:
    raise FileNotFoundError('Image not found. Please place your photo in the images folder and name it your_photo.jpg')
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
noisy_img = cv2.imread('images/produced/q2/noisy_grayscale.jpg', cv2.IMREAD_GRAYSCALE)
if noisy_img is None:
    raise FileNotFoundError('Noisy grayscale image not found. Please run Q2 to generate it.')

# Step 2: Apply Canny edge detector to both images
edges_clean = canny_edge(img)
edges_noisy = canny_edge(noisy_img)

# Step 3: Display and save results plot
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(edges_clean, cmap='gray')
plt.title('Canny Edge (Clean)')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(edges_noisy, cmap='gray')
plt.title('Canny Edge (Noisy)')
plt.axis('off')
plt.tight_layout()
fig.savefig('images/plots/q4_canny_edges.png')  # Save plot for report
plt.show()


# Step 4: Save processed images for report in produced/q4 using OpenCV
cv2.imwrite('images/produced/q4/canny_clean.jpg', edges_clean)
cv2.imwrite('images/produced/q4/canny_noisy.jpg', edges_noisy)