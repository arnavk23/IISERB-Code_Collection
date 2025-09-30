"""
Q2: Add Gaussian noise to RGB image, visualize channels and histograms (noisy and grayscale).

This script demonstrates:
1. Loading an RGB image (your own photo) from disk.
2. Adding Gaussian noise to the image using numpy.
3. Visualizing the noisy image and its individual color channels (R, G, B).
4. Converting the noisy image to grayscale.
5. Computing and displaying histograms for each channel and the grayscale image.
6. Saving all plots and processed images in organized folders for reporting.

All image processing is done from scratch, without OpenCV or PIL.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Load image using OpenCV
img_bgr = cv2.imread('images/your_photo.jpg')  # Reads image as BGR
if img_bgr is None:
    raise FileNotFoundError('Image not found. Please place your photo in the images folder and name it your_photo.jpg')
# Convert BGR to RGB for visualization and processing
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 3: Add Gaussian noise
# Noise parameters: mean=0, variance=15
mean = 0
var = 15
sigma = var ** 0.5
# Generate Gaussian noise for each pixel and channel
noise = np.random.normal(mean, sigma, img_rgb.shape).astype(np.int16)
# Add noise and clip to valid range [0,255]
noisy_img = np.clip(img_rgb + noise, 0, 255).astype(np.uint8)

# Step 4: Split channels for visualization
R = noisy_img[:,:,0]  # Red channel
G = noisy_img[:,:,1]  # Green channel
B = noisy_img[:,:,2]  # Blue channel

# Step 5: Convert noisy image to grayscale using standard luminance formula
gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Step 6: Display and save noisy RGB and channel plots
fig1 = plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(noisy_img)
plt.title('Noisy RGB Image')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(R, cmap='Reds')
plt.title('Noisy Red Channel')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(G, cmap='Greens')
plt.title('Noisy Green Channel')
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(B, cmap='Blues')
plt.title('Noisy Blue Channel')
plt.axis('off')
plt.tight_layout()
fig1.savefig('images/plots/q2_noisy_channels.png')  # Save plot for report
plt.show()

# Step 7: Display and save noisy grayscale image plot
fig2 = plt.figure(figsize=(6,4))
plt.imshow(gray, cmap='gray')
plt.title('Noisy Grayscale Image')
plt.axis('off')
plt.tight_layout()
fig2.savefig('images/plots/q2_noisy_grayscale.png')  # Save plot for report
plt.show()

# Step 8: Compute and save histograms for each channel and grayscale
fig3 = plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.hist(R.ravel(), bins=256, color='red', alpha=0.7)
plt.title('Noisy Red Histogram')
plt.subplot(2,2,2)
plt.hist(G.ravel(), bins=256, color='green', alpha=0.7)
plt.title('Noisy Green Histogram')
plt.subplot(2,2,3)
plt.hist(B.ravel(), bins=256, color='blue', alpha=0.7)
plt.title('Noisy Blue Histogram')
plt.subplot(2,2,4)
plt.hist(gray.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Noisy Grayscale Histogram')
plt.tight_layout()
fig3.savefig('images/plots/q2_noisy_histograms.png')  # Save plot for report
plt.show()


# Step 9: Save processed images for report in produced/q2 using OpenCV
cv2.imwrite('images/produced/q2/noisy_rgb.jpg', cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
cv2.imwrite('images/produced/q2/noisy_grayscale.jpg', gray)