"""
Q1: Read an RGB image, convert to grayscale, visualize R, G, B channels, and display histograms.

This script demonstrates basic image processing operations using only numpy and matplotlib:
1. Loads an RGB image (your own photo) from disk.
2. Converts the image to grayscale using the standard luminance formula.
3. Visualizes the individual color channels (Red, Green, Blue).
4. Computes and displays histograms for each channel and the grayscale image.
5. Saves all plots and processed images in organized folders for reporting.

All image processing is done from scratch, without OpenCV or PIL.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Load your own image using OpenCV (update the path as needed)
img_bgr = cv2.imread('images/your_photo.jpg')  # Reads image as BGR
if img_bgr is None:
    raise FileNotFoundError('Image not found. Please place your photo in the images folder and name it your_photo.jpg')
# Convert BGR to RGB for visualization and processing
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 3: Split channels for visualization
R = img[:,:,0]  # Red channel
G = img[:,:,1]  # Green channel
B = img[:,:,2]  # Blue channel

# Step 4: Convert to grayscale using standard luminance formula
# Y = 0.299*R + 0.587*G + 0.114*B
gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Step 5: Display and save original and grayscale images plot
fig1 = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img.astype(np.uint8))
plt.title('Original RGB Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.tight_layout()
fig1.savefig('images/plots/q1_original_grayscale.png')  # Save plot for report
plt.show()

# Step 6: Visualize and save individual color channels
fig2 = plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(R, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(G, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(B, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')
plt.tight_layout()
fig2.savefig('images/plots/q1_channels.png')  # Save plot for report
plt.show()

# Step 7: Compute and save histograms for each channel and grayscale
fig3 = plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.hist(R.ravel(), bins=256, color='red', alpha=0.7)
plt.title('Red Channel Histogram')
plt.subplot(2,2,2)
plt.hist(G.ravel(), bins=256, color='green', alpha=0.7)
plt.title('Green Channel Histogram')
plt.subplot(2,2,3)
plt.hist(B.ravel(), bins=256, color='blue', alpha=0.7)
plt.title('Blue Channel Histogram')
plt.subplot(2,2,4)
plt.hist(gray.ravel(), bins=256, color='gray', alpha=0.7)
plt.title('Grayscale Histogram')
plt.tight_layout()
fig3.savefig('images/plots/q1_histograms.png')  # Save plot for report
plt.show()


# Step 8: Save processed images for report in produced/q1 using OpenCV
# Convert RGB channels to uint8 before saving
cv2.imwrite('images/produced/q1/grayscale.jpg', gray)
cv2.imwrite('images/produced/q1/red_channel.jpg', R)
cv2.imwrite('images/produced/q1/green_channel.jpg', G)
cv2.imwrite('images/produced/q1/blue_channel.jpg', B)
