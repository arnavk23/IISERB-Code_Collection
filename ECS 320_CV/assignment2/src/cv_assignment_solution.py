"""
Computer Vision Assignment 2 Solution
=====================================
Author: Arnav Kapoor (23060)
Roll Number: 23060
Course: DSE312/DSE602/ECS320 - Computer Vision
Institution: Indian Institute of Science Education and Research Bhopal
Date: September 2025

ASSIGNMENT OVERVIEW:
===================
This comprehensive solution implements all required components for CV Assignment 2:

PART 1: INTEGRAL IMAGE AND HAAR-LIKE FEATURES
- Integral image computation with O(n²) preprocessing and O(1) rectangle queries
- Three types of Haar-like features for pattern detection
- Verification system to ensure correctness of integral image computation
- Application to center 50×50 region with minimum filter size 24×24

PART 2: TEXTURE CLASSIFICATION ON KTH-TIPS DATASET
- Four different feature extraction methods implemented from scratch
- Support Vector Machine (SVM) classification with RBF kernel
- Comprehensive performance analysis and comparison
- Visualization of results and feature representations

FEATURE EXTRACTION METHODS:
===========================
1. Raw Pixel Features: Direct pixel intensity values (baseline method)
2. Local Binary Patterns (LBP): Texture descriptors using local neighborhood
3. Bag-of-Words (BoW): Visual vocabulary from image patches
4. Histogram of Oriented Gradients (HoG): Edge and shape descriptors

TECHNICAL SPECIFICATIONS:
========================
- No specialized CV libraries for core processing (only basic operations allowed)
- Generic implementations that work on any image size
- 70-30 train-test split as required
- Classification for all 10 texture classes
- Extensive error handling and robustness
- Comprehensive visualization and analysis

FILES STRUCTURE:
===============
- cv_assignment_solution.py: Main implementation (this file)
- demo_solution.py: Demo version with subset of data
- test_solution.py: Basic functionality tests
- quick_demo.py: Quick demonstration script
- generate_images.py: Batch image generation utility
"""

# ============================================================================
# IMPORT STATEMENTS AND DEPENDENCIES
# ============================================================================

# Core scientific computing libraries
import numpy as np                    # Numerical operations and array handling
import matplotlib.pyplot as plt       # Plotting and visualization
from matplotlib.patches import Rectangle  # For drawing rectangles on plots

# Machine learning libraries (only SVM classifier as permitted)
from sklearn.svm import SVC                           # Support Vector Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler      # Feature normalization
from sklearn.cluster import KMeans                   # K-means clustering for BoW

# System and file handling libraries
import os                            # Operating system interface
import glob                          # Unix-style pathname pattern expansion
from PIL import Image                # Basic image loading (permitted)
import time                          # Timing operations

# Optional enhanced visualization (graceful degradation if not available)
try:
    import seaborn as sns            # Enhanced statistical plotting
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False              # Fall back to matplotlib only

# ============================================================================
# MAIN CLASS DEFINITION
# ============================================================================

class CVAssignmentSolution:
    """
    Complete Computer Vision Assignment 2 Solution
    ============================================
    
    This class encapsulates all functionality required for the assignment:
    - Integral image computation and verification
    - Haar-like feature extraction (3 types)
    - Multiple feature extraction methods for texture classification
    - SVM classification and performance evaluation
    - Comprehensive visualization and analysis
    
    The implementation follows object-oriented design principles with:
    - Clear separation of concerns
    - Extensive documentation
    - Robust error handling
    - Modular design for easy testing and maintenance
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the CV Assignment Solution
        ===================================
        
        Sets up the solution environment including:
        - Dataset path configuration
        - Class definitions for KTH-TIPS dataset
        - Output directory creation for saving results
        - Initial parameter setup
        
        Args:
            dataset_path (str): Absolute path to KTH-TIPS dataset directory
                              Should contain subdirectories for each texture class
        
        Raises:
            OSError: If dataset path doesn't exist or isn't accessible
        """
        # Store dataset path for later use
        self.dataset_path = dataset_path
        
        # Define the 10 texture classes in KTH-TIPS dataset
        # These correspond to subdirectory names in the dataset
        self.classes = [
            'aluminium_foil',  # Metallic reflective surface
            'brown_bread',     # Organic porous texture  
            'corduroy',        # Fabric with parallel ridges
            'cotton',          # Natural fiber texture
            'cracker',         # Baked food with rough surface
            'linen',           # Woven fabric texture
            'orange_peel',     # Citrus fruit skin texture
            'sandpaper',       # Abrasive paper surface
            'sponge',          # Porous cleaning material
            'styrofoam'        # Synthetic foam texture
        ]
        
        # Create output directory for saving generated images and results
        # This ensures all visualizations are preserved for analysis
        self.output_dir = "output_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Initialize performance tracking (will be populated during analysis)
        self.performance_metrics = {}
        
        # Configuration parameters (can be adjusted for different experiments)
        self.random_seed = 42          # For reproducible results
        self.test_size = 0.3           # 30% for testing (70% for training as required)
        self.svm_kernel = 'rbf'        # Radial Basis Function kernel for SVM
    
    # ============================================================================
    # PART 1: INTEGRAL IMAGE AND HAAR-LIKE FEATURE EXTRACTION
    # ============================================================================
    
    def compute_integral_image(self, image):
        """
        Compute the Integral Image Representation
        ========================================
        
        The integral image (also known as summed-area table) is a data structure
        that allows for fast computation of rectangular region sums. Each pixel
        in the integral image contains the sum of all pixels above and to the left
        of it in the original image.
        
        MATHEMATICAL DEFINITION:
        If I(x,y) is the original image, then the integral image II(x,y) is:
        II(x,y) = Σ(i=0 to x) Σ(j=0 to y) I(i,j)
        
        COMPUTATIONAL ADVANTAGES:
        - Preprocessing: O(n²) where n is image dimension
        - Rectangle sum queries: O(1) constant time
        - Essential for fast Haar-like feature computation
        
        ALGORITHM STEPS:
        1. Handle color images by converting to grayscale
        2. Initialize integral image array with same dimensions
        3. Fill first row using cumulative sum
        4. Fill first column using cumulative sum  
        5. Fill remaining pixels using dynamic programming recurrence:
           II(x,y) = I(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)
        
        Args:
            image (np.ndarray): Input image (2D grayscale or 3D color)
                              Shape: (height, width) or (height, width, channels)
            
        Returns:
            np.ndarray: Integral image with same height and width as input
                       Shape: (height, width)
                       Type: float64 for numerical precision
        
        Example:
            Original:    Integral:
            [1 2 3]  →  [1  3  6 ]
            [4 5 6]     [5 12 21 ]
            [7 8 9]     [12 27 45]
        """
        # Convert color images to grayscale for processing
        # Uses simple averaging across RGB channels
        if len(image.shape) == 3:
            print("Converting color image to grayscale using channel averaging")
            image = np.mean(image, axis=2)
        
        # Get image dimensions
        height, width = image.shape
        print(f"Computing integral image for {height}×{width} image")
        
        # Initialize integral image with zeros
        # Using float64 for numerical precision with large sums
        integral_img = np.zeros((height, width), dtype=np.float64)
        
        # STEP 1: Fill the first row using cumulative sum
        # II(0,j) = Σ(k=0 to j) I(0,k)
        integral_img[0, 0] = image[0, 0]
        for j in range(1, width):
            integral_img[0, j] = integral_img[0, j-1] + image[0, j]
        
        # STEP 2: Fill the first column using cumulative sum
        # II(i,0) = Σ(k=0 to i) I(k,0)
        for i in range(1, height):
            integral_img[i, 0] = integral_img[i-1, 0] + image[i, 0]
        
        # STEP 3: Fill remaining pixels using dynamic programming
        # Recurrence relation: II(i,j) = I(i,j) + II(i-1,j) + II(i,j-1) - II(i-1,j-1)
        # The subtraction is needed because II(i-1,j-1) is counted twice
        for i in range(1, height):
            for j in range(1, width):
                integral_img[i, j] = (
                    image[i, j] +              # Current pixel value
                    integral_img[i-1, j] +     # Sum above current pixel
                    integral_img[i, j-1] -     # Sum to left of current pixel  
                    integral_img[i-1, j-1]     # Subtract overlap (counted twice)
                )
        
        print(f"Integral image computation completed")
        return integral_img
    
    def get_rectangle_sum(self, integral_img, x1, y1, x2, y2):
        """
        Get Sum of Pixel Values in Rectangle using Integral Image
        =======================================================
        
        This function demonstrates the power of integral images by computing
        the sum of all pixel values in any rectangular region in O(1) time,
        regardless of the rectangle size.
        
        MATHEMATICAL PRINCIPLE:
        To find sum of rectangle from (x1,y1) to (x2,y2):
        Sum = II(x2,y2) - II(x1-1,y2) - II(x2,y1-1) + II(x1-1,y1-1)
        
        VISUAL EXPLANATION:
        Consider rectangle R in integral image:
        ┌─────┬─────┐
        │  A  │  B  │
        ├─────┼─────┤
        │  C  │  R  │
        └─────┴─────┘
        
        Where:
        - A = II(x1-1, y1-1)
        - B = II(x1-1, y2)  
        - C = II(x2, y1-1)
        - Total = II(x2, y2)
        
        Rectangle sum = Total - B - C + A
        (A is added back because it's subtracted twice)
        
        EDGE CASES HANDLED:
        - Top-left corner at origin (0,0)
        - Rectangle touching left edge (x1=0)  
        - Rectangle touching top edge (y1=0)
        - Combinations of above cases
        
        Args:
            integral_img (np.ndarray): Precomputed integral image
                                     Shape: (height, width)
            x1, y1 (int): Top-left corner coordinates (inclusive)
                         Must satisfy: 0 ≤ x1 ≤ x2 < height
                                      0 ≤ y1 ≤ y2 < width  
            x2, y2 (int): Bottom-right corner coordinates (inclusive)
            
        Returns:
            float: Sum of all pixel values in the specified rectangle
            
        Raises:
            IndexError: If coordinates are outside image bounds
            
        Time Complexity: O(1) - constant time regardless of rectangle size
        Space Complexity: O(1) - no additional storage needed
        """
        # Validate input coordinates
        if x1 < 0 or y1 < 0 or x2 >= integral_img.shape[0] or y2 >= integral_img.shape[1]:
            raise IndexError(f"Rectangle coordinates ({x1},{y1}) to ({x2},{y2}) "
                           f"are outside image bounds {integral_img.shape}")
        
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid rectangle: top-left ({x1},{y1}) must be "
                           f"above and left of bottom-right ({x2},{y2})")
        
        # Handle different edge cases for optimal computation
        if x1 == 0 and y1 == 0:
            # Rectangle starts at origin - simplest case
            return integral_img[x2, y2]
            
        elif x1 == 0:
            # Rectangle touches left edge - only subtract top area
            return integral_img[x2, y2] - integral_img[x2, y1-1]
            
        elif y1 == 0:
            # Rectangle touches top edge - only subtract left area  
            return integral_img[x2, y2] - integral_img[x1-1, y2]
            
        else:
            # General case - use full formula
            return (integral_img[x2, y2] -           # Total area including rectangle
                   integral_img[x1-1, y2] -         # Subtract left area
                   integral_img[x2, y1-1] +         # Subtract top area
                   integral_img[x1-1, y1-1])        # Add back overlap (subtracted twice)
    
    def verify_integral_image(self, original_img, integral_img, x1, y1, x2, y2):
        """
        Verify Integral Image Correctness by Comparing with Direct Summation
        ===================================================================
        
        This verification function ensures the integral image implementation
        is correct by comparing results with brute-force direct summation.
        It's essential for validating the fundamental data structure before
        using it for Haar feature computation.
        
        VERIFICATION PROCESS:
        1. Compute rectangle sum using integral image (O(1) time)
        2. Compute same rectangle sum using direct pixel summation (O(n²) time)
        3. Compare results within floating-point precision tolerance
        4. Return detailed comparison for analysis
        
        NUMERICAL CONSIDERATIONS:
        - Uses tolerance of 1e-10 for floating-point comparison
        - Handles potential rounding errors from repeated additions
        - Converts color images to grayscale consistently
        
        Args:
            original_img (np.ndarray): Original image before integral transformation
                                     Shape: (height, width) or (height, width, channels)
            integral_img (np.ndarray): Computed integral image 
                                     Shape: (height, width)
            x1, y1, x2, y2 (int): Rectangle coordinates to verify
                                 Top-left: (x1, y1), Bottom-right: (x2, y2)
            
        Returns:
            tuple: Three-element tuple containing:
                - integral_sum (float): Sum computed using integral image
                - direct_sum (float): Sum computed using direct summation  
                - is_correct (bool): True if sums match within tolerance
                
        Example:
            For rectangle (10,10) to (20,20):
            integral_sum, direct_sum, is_correct = verify_integral_image(...)
            if is_correct:
                print("Integral image implementation verified!")
        """
        # STEP 1: Get sum using fast integral image method
        integral_sum = self.get_rectangle_sum(integral_img, x1, y1, x2, y2)
        
        # STEP 2: Calculate direct sum by iterating through all pixels
        # Ensure consistency with integral image preprocessing
        if len(original_img.shape) == 3:
            # Convert to grayscale using same method as integral image computation
            original_img = np.mean(original_img, axis=2)
        
        # Direct summation over the specified rectangle region
        # Note: slice notation [x1:x2+1, y1:y2+1] includes both endpoints
        direct_sum = np.sum(original_img[x1:x2+1, y1:y2+1])
        
        # STEP 3: Compare results within floating-point precision tolerance
        # Using absolute difference to handle potential rounding errors
        tolerance = 1e-10  # Very strict tolerance for numerical precision
        is_correct = np.abs(integral_sum - direct_sum) < tolerance
        
        # Log verification details for debugging
        if not is_correct:
            print(f"WARNING: Integral image verification failed!")
            print(f"  Rectangle: ({x1},{y1}) to ({x2},{y2})")
            print(f"  Integral sum: {integral_sum}")
            print(f"  Direct sum: {direct_sum}")
            print(f"  Difference: {abs(integral_sum - direct_sum)}")
        
        return integral_sum, direct_sum, is_correct
    
    def haar_feature_type1(self, integral_img, x, y, w, h):
        """
        Compute Haar-like Feature Type 1: Left-Right Pattern
        ===================================================
        
        This function implements the first type of Haar-like feature that
        detects vertical edges and left-right intensity differences. It's
        fundamental for object detection and pattern recognition tasks.
        
        PATTERN DESCRIPTION:
        ┌─────────┬─────────┐
        │  LEFT   │  RIGHT  │  ← Feature window
        │ (light) │ (dark)  │
        └─────────┴─────────┘
        
        MATHEMATICAL DEFINITION:
        Feature Value = Sum(Left Rectangle) - Sum(Right Rectangle)
        
        DETECTION CAPABILITIES:
        - Vertical edges (light-to-dark or dark-to-light transitions)
        - Object boundaries with different illumination
        - Texture patterns with left-right asymmetry
        - Facial features (e.g., nose bridge, eye regions)
        
        APPLICATIONS IN COMPUTER VISION:
        - Face detection (Viola-Jones algorithm)
        - Object recognition and classification
        - Texture analysis and segmentation
        - Edge detection at multiple scales
        
        Args:
            integral_img (np.ndarray): Precomputed integral image
                                     Shape: (height, width)
            x, y (int): Top-left corner of feature window
                       Coordinates: (row, column)
            w, h (int): Width and height of feature window
                       Must satisfy: w ≥ 2 (for left-right split)
                                    x+h ≤ image_height
                                    y+w ≤ image_width
            
        Returns:
            float: Haar feature value (positive indicates left > right intensity)
                  Range: Depends on image values and window size
                  
        Raises:
            ValueError: If window dimensions don't allow proper splitting
            IndexError: If feature window extends beyond image boundaries
        """
        # Validate input parameters
        if w < 2:
            raise ValueError(f"Width {w} too small for left-right split (minimum: 2)")
        
        # Calculate split point for left and right rectangles
        mid_w = w // 2  # Integer division ensures clean split
        
        # COMPUTE LEFT RECTANGLE SUM
        # Rectangle from (x, y) to (x+h-1, y+mid_w-1)
        left_sum = self.get_rectangle_sum(integral_img, x, y, x+h-1, y+mid_w-1)
        
        # COMPUTE RIGHT RECTANGLE SUM  
        # Rectangle from (x, y+mid_w) to (x+h-1, y+w-1)
        right_sum = self.get_rectangle_sum(integral_img, x, y+mid_w, x+h-1, y+w-1)
        
        # COMPUTE FEATURE VALUE
        # Positive value: left side brighter than right side
        # Negative value: right side brighter than left side
        feature_value = left_sum - right_sum
        
        return feature_value
    
    def haar_feature_type2(self, integral_img, x, y, w, h):
        """
        Compute Haar-like Feature Type 2: Top-Bottom Pattern
        ===================================================
        
        This function implements the second type of Haar-like feature that
        detects horizontal edges and top-bottom intensity differences.
        
        PATTERN DESCRIPTION:
        ┌─────────────┐
        │    TOP      │  ← Light region
        │  (light)    │
        ├─────────────┤
        │   BOTTOM    │  ← Dark region  
        │   (dark)    │
        └─────────────┘
        
        MATHEMATICAL DEFINITION:
        Feature Value = Sum(Top Rectangle) - Sum(Bottom Rectangle)
        
        Args:
            integral_img (np.ndarray): Integral image
            x, y (int): Top-left position
            w, h (int): Width and height of feature window
            
        Returns:
            float: Haar feature value
        """
        mid_h = h // 2
        
        # Top rectangle sum
        top_sum = self.get_rectangle_sum(integral_img, x, y, x+mid_h-1, y+w-1)
        
        # Bottom rectangle sum
        bottom_sum = self.get_rectangle_sum(integral_img, x+mid_h, y, x+h-1, y+w-1)
        
        return top_sum - bottom_sum
    
    def haar_feature_type3(self, integral_img, x, y, w, h):
        """
        Compute Haar-like feature Type 3: Three-rectangle pattern
        (Left rectangle + Right rectangle - Middle rectangle)
        
        Args:
            integral_img (np.ndarray): Integral image
            x, y (int): Top-left position
            w, h (int): Width and height of feature window
            
        Returns:
            float: Haar feature value
        """
        third_w = w // 3
        
        # Left rectangle sum
        left_sum = self.get_rectangle_sum(integral_img, x, y, x+h-1, y+third_w-1)
        
        # Middle rectangle sum
        middle_sum = self.get_rectangle_sum(integral_img, x, y+third_w, x+h-1, y+2*third_w-1)
        
        # Right rectangle sum
        right_sum = self.get_rectangle_sum(integral_img, x, y+2*third_w, x+h-1, y+w-1)
        
        return left_sum + right_sum - middle_sum
    
    def extract_haar_features(self, image, region_x, region_y, region_size, min_filter_size=24):
        """
        Extract Haar-like features from a specific region of the image.
        
        Args:
            image (np.ndarray): Input image
            region_x, region_y (int): Top-left corner of region
            region_size (int): Size of the region
            min_filter_size (int): Minimum filter size
            
        Returns:
            dict: Dictionary containing features for each Haar type
        """
        # Compute integral image
        integral_img = self.compute_integral_image(image)
        
        features = {'type1': [], 'type2': [], 'type3': []}
        
        # Extract features with different filter sizes
        for filter_size in range(min_filter_size, region_size + 1, 4):  # Step by 4 for efficiency
            for x in range(region_x, region_x + region_size - filter_size + 1, 4):
                for y in range(region_y, region_y + region_size - filter_size + 1, 4):
                    # Type 1: Left-Right
                    if filter_size % 2 == 0:  # Ensure even width for left-right split
                        feat1 = self.haar_feature_type1(integral_img, x, y, filter_size, filter_size)
                        features['type1'].append(feat1)
                    
                    # Type 2: Top-Bottom
                    if filter_size % 2 == 0:  # Ensure even height for top-bottom split
                        feat2 = self.haar_feature_type2(integral_img, x, y, filter_size, filter_size)
                        features['type2'].append(feat2)
                    
                    # Type 3: Three-rectangle
                    if filter_size % 3 == 0:  # Ensure divisible by 3 for three rectangles
                        feat3 = self.haar_feature_type3(integral_img, x, y, filter_size, filter_size)
                        features['type3'].append(feat3)
        
        return features
    
    # ==================== PART 2: TEXTURE CLASSIFICATION ====================
    
    def load_dataset(self):
        """
        Load the KTH-TIPS dataset.
        
        Returns:
            tuple: (images, labels, label_names)
        """
        images = []
        labels = []
        label_names = []
        
        print("Loading KTH-TIPS dataset...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.exists(class_path):
                image_files = glob.glob(os.path.join(class_path, "*.png"))
                print(f"Loading {len(image_files)} images from {class_name}")
                
                for img_file in image_files:
                    try:
                        img = Image.open(img_file)
                        img_array = np.array(img)
                        
                        # Convert to grayscale if needed
                        if len(img_array.shape) == 3:
                            img_array = np.mean(img_array, axis=2)
                        
                        # Resize to standard size to ensure all images have same dimensions
                        img_pil = Image.fromarray(img_array.astype(np.uint8))
                        img_resized = img_pil.resize((200, 200))  # Standard size
                        img_array = np.array(img_resized)
                        
                        images.append(img_array)
                        labels.append(class_idx)
                        label_names.append(class_name)
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
        
        print(f"Total images loaded: {len(images)}")
        return np.array(images), np.array(labels), label_names
    
    def extract_raw_pixel_features(self, images, target_size=(64, 64)):
        """
        Extract raw pixel intensity features by resizing images.
        
        Args:
            images (np.ndarray): Array of images
            target_size (tuple): Target size for resizing
            
        Returns:
            np.ndarray: Feature matrix
        """
        features = []
        
        for img in images:
            # Resize image
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_resized = img_pil.resize(target_size)
            img_array = np.array(img_resized)
            
            # Flatten to 1D feature vector
            features.append(img_array.flatten())
        
        return np.array(features)
    
    def compute_lbp(self, image, radius=1, n_points=8):
        """
        Compute Local Binary Pattern for an image.
        
        Args:
            image (np.ndarray): Input grayscale image
            radius (int): Radius of circle
            n_points (int): Number of points on circle
            
        Returns:
            np.ndarray: LBP image
        """
        height, width = image.shape
        lbp_image = np.zeros((height, width), dtype=np.uint8)
        
        # Create circular neighbor offsets
        angles = 2 * np.pi * np.arange(n_points) / n_points
        dy = -radius * np.sin(angles)
        dx = radius * np.cos(angles)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center_pixel = image[i, j]
                binary_string = ""
                
                for k in range(n_points):
                    # Calculate neighbor coordinates
                    ny = i + dy[k]
                    nx = j + dx[k]
                    
                    # Bilinear interpolation for non-integer coordinates
                    y1, x1 = int(ny), int(nx)
                    y2, x2 = y1 + 1, x1 + 1
                    
                    # Ensure bounds
                    if y2 >= height or x2 >= width or y1 < 0 or x1 < 0:
                        neighbor_value = center_pixel
                    else:
                        # Bilinear interpolation
                        wa = (x2 - nx) * (y2 - ny)
                        wb = (nx - x1) * (y2 - ny)
                        wc = (x2 - nx) * (ny - y1)
                        wd = (nx - x1) * (ny - y1)
                        
                        neighbor_value = (wa * image[y1, x1] + wb * image[y1, x2] + 
                                        wc * image[y2, x1] + wd * image[y2, x2])
                    
                    # Compare with center pixel
                    if neighbor_value >= center_pixel:
                        binary_string += "1"
                    else:
                        binary_string += "0"
                
                # Convert binary string to decimal
                lbp_image[i, j] = int(binary_string, 2)
        
        return lbp_image
    
    def extract_lbp_features(self, images):
        """
        Extract LBP histogram features from images.
        
        Args:
            images (np.ndarray): Array of images
            
        Returns:
            np.ndarray: LBP feature matrix
        """
        features = []
        
        for img in images:
            # Compute LBP
            lbp_img = self.compute_lbp(img)
            
            # Compute histogram
            hist, _ = np.histogram(lbp_img.flatten(), bins=256, range=(0, 256))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)  # Avoid division by zero
            
            features.append(hist)
        
        return np.array(features)
    
    def extract_bow_features(self, images, n_clusters=100, patch_size=16):
        """
        Extract Bag-of-Words features using image patches.
        
        Args:
            images (np.ndarray): Array of images
            n_clusters (int): Number of clusters for vocabulary
            patch_size (int): Size of patches
            
        Returns:
            np.ndarray: BoW feature matrix
        """
        print("Extracting patches for BoW...")
        
        # Extract patches from all images
        all_patches = []
        for img in images:
            patches = self.extract_patches(img, patch_size)
            all_patches.extend(patches)
        
        all_patches = np.array(all_patches)
        print(f"Total patches extracted: {len(all_patches)}")
        
        # Learn vocabulary using K-means
        print("Learning vocabulary...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_patches)
        
        # Extract BoW features for each image
        print("Extracting BoW features...")
        features = []
        for img in images:
            patches = self.extract_patches(img, patch_size)
            if len(patches) == 0:
                # Handle case where no patches could be extracted
                bow_hist = np.zeros(n_clusters)
            else:
                patches = np.array(patches)
                labels = kmeans.predict(patches)
                bow_hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))
                bow_hist = bow_hist.astype(np.float32)
                bow_hist /= (bow_hist.sum() + 1e-7)  # Normalize
            
            features.append(bow_hist)
        
        return np.array(features)
    
    def extract_patches(self, image, patch_size):
        """
        Extract patches from an image.
        
        Args:
            image (np.ndarray): Input image
            patch_size (int): Size of patches
            
        Returns:
            list: List of flattened patches
        """
        patches = []
        height, width = image.shape
        
        for i in range(0, height - patch_size + 1, patch_size // 2):  # Overlap patches
            for j in range(0, width - patch_size + 1, patch_size // 2):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        
        return patches
    
    def compute_gradients(self, image):
        """
        Compute gradients of an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (gradient_magnitude, gradient_direction)
        """
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Pad image
        padded = np.pad(image, 1, mode='edge')
        
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        # Compute gradients
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                grad_x[i, j] = np.sum(padded[i:i+3, j:j+3] * sobel_x)
                grad_y[i, j] = np.sum(padded[i:i+3, j:j+3] * sobel_y)
        
        # Compute magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
        direction[direction < 0] += 180  # Convert to 0-180 range
        
        return magnitude, direction
    
    def extract_hog_features(self, images, cell_size=8, block_size=2, n_bins=9):
        """
        Extract Histogram of Oriented Gradients (HoG) features.
        
        Args:
            images (np.ndarray): Array of images
            cell_size (int): Size of cells
            block_size (int): Size of blocks (in cells)
            n_bins (int): Number of orientation bins
            
        Returns:
            np.ndarray: HoG feature matrix
        """
        features = []
        
        for img in images:
            # Resize image to ensure consistent dimensions
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_resized = img_pil.resize((128, 128))  # Standard size for HoG
            img_array = np.array(img_resized)
            
            # Compute gradients
            magnitude, direction = self.compute_gradients(img_array)
            
            # Compute HoG features
            hog_feature = self.compute_hog_descriptor(magnitude, direction, cell_size, block_size, n_bins)
            features.append(hog_feature)
        
        return np.array(features)
    
    def compute_hog_descriptor(self, magnitude, direction, cell_size, block_size, n_bins):
        """
        Compute HoG descriptor for an image.
        
        Args:
            magnitude (np.ndarray): Gradient magnitude
            direction (np.ndarray): Gradient direction
            cell_size (int): Size of cells
            block_size (int): Size of blocks
            n_bins (int): Number of bins
            
        Returns:
            np.ndarray: HoG feature vector
        """
        height, width = magnitude.shape
        
        # Calculate number of cells
        n_cells_y = height // cell_size
        n_cells_x = width // cell_size
        
        # Initialize cell histograms
        cell_histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
        
        # Compute histograms for each cell
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extract cell region
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_dir = direction[y_start:y_end, x_start:x_end]
                
                # Compute histogram
                hist = np.zeros(n_bins)
                bin_width = 180.0 / n_bins
                
                for y in range(cell_size):
                    for x in range(cell_size):
                        angle = cell_dir[y, x]
                        mag = cell_mag[y, x]
                        
                        # Find bins for interpolation
                        bin_idx = angle / bin_width
                        bin1 = int(bin_idx) % n_bins
                        bin2 = (bin1 + 1) % n_bins
                        
                        # Interpolate between bins
                        frac = bin_idx - bin1
                        hist[bin1] += mag * (1 - frac)
                        hist[bin2] += mag * frac
                
                cell_histograms[i, j, :] = hist
        
        # Normalize blocks
        hog_features = []
        for i in range(n_cells_y - block_size + 1):
            for j in range(n_cells_x - block_size + 1):
                # Extract block
                block = cell_histograms[i:i+block_size, j:j+block_size, :].flatten()
                
                # L2 normalization
                norm = np.linalg.norm(block)
                if norm > 0:
                    block = block / norm
                
                hog_features.extend(block)
        
        return np.array(hog_features)
    
    def visualize_hog(self, image, cell_size=8):
        """
        Visualize HoG features by showing gradient orientations.
        
        Args:
            image (np.ndarray): Input image
            cell_size (int): Size of cells
            
        Returns:
            tuple: (hog_image, magnitude, direction)
        """
        # Compute gradients
        magnitude, direction = self.compute_gradients(image)
        
        height, width = image.shape
        n_cells_y = height // cell_size
        n_cells_x = width // cell_size
        
        # Create HoG visualization
        hog_image = np.zeros((n_cells_y * cell_size, n_cells_x * cell_size))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extract cell region
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_dir = direction[y_start:y_end, x_start:x_end]
                
                # Find dominant direction
                mean_direction = np.mean(cell_dir)
                mean_magnitude = np.mean(cell_mag)
                
                # Draw line representing dominant gradient
                center_y, center_x = i * cell_size + cell_size // 2, j * cell_size + cell_size // 2
                
                # Convert direction to line endpoints
                angle_rad = mean_direction * np.pi / 180
                line_length = min(cell_size // 2, int(mean_magnitude / 10))
                
                end_y = center_y + line_length * np.sin(angle_rad)
                end_x = center_x + line_length * np.cos(angle_rad)
                
                # Draw line (simplified visualization)
                try:
                    y_indices = np.linspace(center_y, end_y, line_length).astype(int)
                    x_indices = np.linspace(center_x, end_x, line_length).astype(int)
                    
                    # Ensure indices are within bounds
                    valid_mask = ((y_indices >= 0) & (y_indices < hog_image.shape[0]) & 
                                 (x_indices >= 0) & (x_indices < hog_image.shape[1]))
                    
                    hog_image[y_indices[valid_mask], x_indices[valid_mask]] = mean_magnitude
                except:
                    pass  # Skip if line drawing fails
        
        return hog_image, magnitude, direction
    
    def save_sample_images(self, images, labels, n_samples=3):
        """
        Save sample images from each class.
        
        Args:
            images (np.ndarray): Array of images
            labels (np.ndarray): Array of labels
            n_samples (int): Number of samples per class to save
        """
        print("Saving sample images from each class...")
        
        for class_idx, class_name in enumerate(self.classes):
            class_mask = labels == class_idx
            class_images = images[class_mask][:n_samples]
            
            for i, img in enumerate(class_images):
                filename = f"{self.output_dir}/sample_{class_name}_{i+1}.png"
                plt.figure(figsize=(4, 4))
                plt.imshow(img, cmap='gray')
                plt.title(f'{class_name} - Sample {i+1}')
                plt.axis('off')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"Saved {len(self.classes) * n_samples} sample images to {self.output_dir}/")
    
    def save_feature_examples(self, images, labels):
        """
        Save examples of different feature extractions on sample images.
        
        Args:
            images (np.ndarray): Array of images
            labels (np.ndarray): Array of labels
        """
        print("Saving feature extraction examples...")
        
        # Get one sample from first class
        sample_img = images[labels == 0][0]
        
        # LBP example
        lbp_img = self.compute_lbp(sample_img)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(sample_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(lbp_img, cmap='gray')
        plt.title('LBP Image')
        plt.axis('off')
        
        # HoG example
        hog_img, magnitude, direction = self.visualize_hog(sample_img)
        
        plt.subplot(1, 3, 3)
        plt.imshow(magnitude, cmap='hot')
        plt.title('Gradient Magnitude')
        plt.axis('off')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/feature_extraction_examples.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature extraction examples: {filename}")
    
    def save_results_summary(self, all_results):
        """
        Save a summary figure with all results.
        
        Args:
            all_results (list): List of result dictionaries
        """
        print("Creating and saving results summary...")
        
        methods = [result['method'] for result in all_results]
        accuracies = [result['accuracy'] for result in all_results]
        train_times = [result['training_time'] for result in all_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        bars1 = ax1.bar(methods, accuracies, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        bars2 = ax2.bar(methods, train_times, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Performance ranking
        ranking_data = sorted(zip(methods, accuracies), key=lambda x: x[1], reverse=True)
        ranked_methods = [item[0] for item in ranking_data]
        ranked_accuracies = [item[1] for item in ranking_data]
        
        bars3 = ax3.barh(range(len(ranked_methods)), ranked_accuracies, 
                        color=['gold', 'silver', '#cd7f32', 'lightgray'])
        ax3.set_title('Performance Ranking', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Accuracy')
        ax3.set_yticks(range(len(ranked_methods)))
        # Set y-tick labels properly
        ax3.set_yticklabels(ranked_methods)
        ax3.grid(axis='x', alpha=0.3)
        
        # Add ranking numbers
        for i, acc in enumerate(ranked_accuracies):
            ax3.text(acc + 0.01, i, f'#{i+1}: {acc:.3f}', 
                    va='center', fontweight='bold')
        
        # Summary statistics
        ax4.axis('off')
        summary_text = f"""
ASSIGNMENT SUMMARY

Dataset: KTH-TIPS (810 images, 10 classes)
Train-Test Split: 70-30

BEST PERFORMING METHOD:
{ranked_methods[0]}: {ranked_accuracies[0]:.1%} accuracy

METHODS IMPLEMENTED:
✓ Raw Pixels (baseline)
✓ Local Binary Patterns (texture)
✓ Bag-of-Words (patches)
✓ Histogram of Gradients (edges)

INTEGRAL IMAGE & HAAR FEATURES:
✓ 3 Haar feature types implemented
✓ Fast rectangular sum computation
✓ Applied to 50×50 center region

Student: Arnav Kapoor (23060)
Course: Computer Vision Assignment 2
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        filename = f"{self.output_dir}/complete_results_summary.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved complete results summary: {filename}")
    
    def train_and_evaluate_classifier(self, features, labels, method_name):
        """
        Train and evaluate SVM classifier.
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Labels
            method_name (str): Name of the method
            
        Returns:
            dict: Results dictionary
        """
        print(f"\n=== Training {method_name} classifier ===")
        
        # Split data (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM classifier
        start_time = time.time()
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = svm.predict(X_test_scaled)
        prediction_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Prediction time: {prediction_time:.4f} seconds")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)
        
        results = {
            'method': method_name,
            'accuracy': accuracy,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, method_name):
        """
        Plot confusion matrix.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            method_name (str): Method name for title
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.classes, yticklabels=self.classes)
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            plt.xticks(range(len(self.classes)), self.classes, rotation=45)
            plt.yticks(range(len(self.classes)), self.classes, rotation=0)
        plt.title(f'Confusion Matrix - {method_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the confusion matrix
        filename = f"{self.output_dir}/confusion_matrix_{method_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {filename}")
        
        plt.show()
    
    def create_cameraman_image(self):
        """
        Create a simple cameraman-like test image for Haar feature demonstration.
        
        Returns:
            np.ndarray: Test image
        """
        # Create a 256x256 test image with some patterns
        image = np.zeros((256, 256))
        
        # Add some rectangular patterns
        image[50:150, 50:100] = 100  # Left rectangle
        image[50:150, 150:200] = 200  # Right rectangle
        image[100:200, 100:150] = 150  # Bottom rectangle
        
        # Add some noise
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255)
        
        return image
    
    def run_complete_analysis(self):
        """
        Run the complete CV assignment analysis.
        """
        print("=== Computer Vision Assignment 2 Solution ===")
        print("Author: Arnav Kapoor (23060)")
        
        # Part 1: Integral Image and Haar Features
        print("\n" + "="*50)
        print("PART 1: INTEGRAL IMAGE AND HAAR FEATURES")
        print("="*50)
        
        # Create test image
        test_image = self.create_cameraman_image()
        
        # Test integral image
        print("\n1. Testing Integral Image Implementation:")
        integral_img = self.compute_integral_image(test_image)
        
        # Verify with random rectangles
        for i in range(5):
            x1, y1 = np.random.randint(0, 100, 2)
            x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
            x2, y2 = min(x2, test_image.shape[0]-1), min(y2, test_image.shape[1]-1)
            
            integral_sum, direct_sum, is_correct = self.verify_integral_image(
                test_image, integral_img, x1, y1, x2, y2
            )
            
            print(f"Test {i+1}: Rectangle ({x1},{y1}) to ({x2},{y2})")
            print(f"  Integral sum: {integral_sum:.2f}")
            print(f"  Direct sum: {direct_sum:.2f}")
            print(f"  Correct: {is_correct}")
        
        # Extract Haar features from center 50x50 region
        print("\n2. Extracting Haar Features from center 50x50 region:")
        center_x, center_y = (256 - 50) // 2, (256 - 50) // 2
        haar_features = self.extract_haar_features(
            test_image, center_x, center_y, 50, min_filter_size=24
        )
        
        print(f"Haar Type 1 features: {len(haar_features['type1'])}")
        print(f"Haar Type 2 features: {len(haar_features['type2'])}")
        print(f"Haar Type 3 features: {len(haar_features['type3'])}")
        
        # Visualize test image and some results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title('Test Image')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(integral_img, cmap='viridis')
        plt.title('Integral Image')
        plt.colorbar()
        
        # Show region where Haar features were extracted
        plt.subplot(1, 3, 3)
        plt.imshow(test_image, cmap='gray')
        rect = Rectangle((center_y, center_x), 50, 50, 
                       linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.title('Haar Feature Extraction Region (50x50)')
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{self.output_dir}/haar_features_demo.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Haar features demo: {filename}")
        
        plt.show()
        
        print("\nHaar Feature Analysis:")
        print("- Type 1 (Left-Right): Detects vertical edges and patterns")
        print("- Type 2 (Top-Bottom): Detects horizontal edges and patterns") 
        print("- Type 3 (Three-rectangle): Detects line patterns and textures")
        print("These features are useful in object detection for:")
        print("  * Edge detection at multiple scales")
        print("  * Pattern recognition (eyes, mouth in face detection)")
        print("  * Texture analysis")
        print("  * Fast computation using integral images")
        
        # Part 2: Texture Classification
        print("\n" + "="*50)
        print("PART 2: TEXTURE CLASSIFICATION")
        print("="*50)
        
        # Load dataset
        images, labels, label_names = self.load_dataset()
        
        print(f"\nDataset loaded: {len(images)} images from {len(self.classes)} classes")
        for i, class_name in enumerate(self.classes):
            count = np.sum(labels == i)
            print(f"  {class_name}: {count} images")
        
        # Save sample images from dataset
        self.save_sample_images(images, labels, n_samples=3)
        
        # Save feature extraction examples
        self.save_feature_examples(images, labels)
        
        # Store all results
        all_results = []
        
        # 1. Raw Pixel Classification
        print("\n" + "-"*30)
        print("1. RAW PIXEL CLASSIFICATION")
        print("-"*30)
        
        raw_features = self.extract_raw_pixel_features(images)
        raw_results = self.train_and_evaluate_classifier(raw_features, labels, "Raw Pixels")
        all_results.append(raw_results)
        self.plot_confusion_matrix(raw_results['y_test'], raw_results['y_pred'], "Raw Pixels")
        
        print("\nLimitations of raw pixel representation:")
        print("- Sensitive to illumination changes")
        print("- No translation invariance")
        print("- High dimensional but low semantic content")
        print("- Ignores spatial relationships")
        
        # 2. LBP Classification
        print("\n" + "-"*30)
        print("2. LOCAL BINARY PATTERN CLASSIFICATION")
        print("-"*30)
        
        lbp_features = self.extract_lbp_features(images)
        lbp_results = self.train_and_evaluate_classifier(lbp_features, labels, "LBP")
        all_results.append(lbp_results)
        self.plot_confusion_matrix(lbp_results['y_test'], lbp_results['y_pred'], "LBP")
        
        # Visualize LBP for sample images
        plt.figure(figsize=(15, 10))
        for i in range(min(5, len(images))):
            img = images[i]
            lbp_img = self.compute_lbp(img)
            
            plt.subplot(2, 5, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Original: {self.classes[labels[i]]}')
            plt.axis('off')
            
            plt.subplot(2, 5, i+6)
            plt.imshow(lbp_img, cmap='gray')
            plt.title('LBP')
            plt.axis('off')
        
        plt.suptitle('LBP Visualization for Sample Images')
        plt.tight_layout()
        
        # Save LBP visualization
        filename = f"{self.output_dir}/lbp_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved LBP visualization: {filename}")
        
        plt.show()
        
        # 3. Bag-of-Words Classification
        print("\n" + "-"*30)
        print("3. BAG-OF-WORDS CLASSIFICATION")
        print("-"*30)
        
        bow_features = self.extract_bow_features(images, n_clusters=50)  # Reduced for efficiency
        bow_results = self.train_and_evaluate_classifier(bow_features, labels, "Bag-of-Words")
        all_results.append(bow_results)
        self.plot_confusion_matrix(bow_results['y_test'], bow_results['y_pred'], "Bag-of-Words")
        
        # 4. HoG Classification
        print("\n" + "-"*30)
        print("4. HOG CLASSIFICATION")
        print("-"*30)
        
        hog_features = self.extract_hog_features(images)
        hog_results = self.train_and_evaluate_classifier(hog_features, labels, "HoG")
        all_results.append(hog_results)
        self.plot_confusion_matrix(hog_results['y_test'], hog_results['y_pred'], "HoG")
        
        # Visualize HoG for sample images
        plt.figure(figsize=(15, 10))
        for i in range(min(3, len(images))):
            img = images[i]
            hog_img, magnitude, direction = self.visualize_hog(img)
            
            plt.subplot(3, 3, i*3+1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Original: {self.classes[labels[i]]}')
            plt.axis('off')
            
            plt.subplot(3, 3, i*3+2)
            plt.imshow(magnitude, cmap='hot')
            plt.title('Gradient Magnitude')
            plt.axis('off')
            
            plt.subplot(3, 3, i*3+3)
            plt.imshow(hog_img, cmap='gray')
            plt.title('HoG Visualization')
            plt.axis('off')
        
        plt.suptitle('HoG Feature Visualization')
        plt.tight_layout()
        
        # Save HoG visualization
        filename = f"{self.output_dir}/hog_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved HoG visualization: {filename}")
        
        plt.show()
        
        # Performance Comparison
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        
        # Create comparison table
        methods = []
        accuracies = []
        train_times = []
        pred_times = []
        
        for result in all_results:
            methods.append(result['method'])
            accuracies.append(result['accuracy'])
            train_times.append(result['training_time'])
            pred_times.append(result['prediction_time'])
        
        # Print comparison table
        print(f"{'Method':<15} {'Accuracy':<10} {'Train Time':<12} {'Pred Time':<12}")
        print("-" * 55)
        for i in range(len(methods)):
            print(f"{methods[i]:<15} {accuracies[i]:<10.4f} {train_times[i]:<12.2f} {pred_times[i]:<12.4f}")
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.bar(methods, accuracies)
        plt.title('Classification Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        plt.bar(methods, train_times)
        plt.title('Training Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        plt.bar(methods, pred_times)
        plt.title('Prediction Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save performance comparison
        filename = f"{self.output_dir}/performance_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison: {filename}")
        
        plt.show()
        
        # Analysis and conclusions
        print("\n" + "="*50)
        print("ANALYSIS AND CONCLUSIONS")
        print("="*50)
        
        best_accuracy_idx = np.argmax(accuracies)
        fastest_train_idx = np.argmin(train_times)
        fastest_pred_idx = np.argmin(pred_times)
        
        print(f"\nBest accuracy: {methods[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.4f})")
        print(f"Fastest training: {methods[fastest_train_idx]} ({train_times[fastest_train_idx]:.2f}s)")
        print(f"Fastest prediction: {methods[fastest_pred_idx]} ({pred_times[fastest_pred_idx]:.4f}s)")
        
        # Save comprehensive results summary
        self.save_results_summary(all_results)
        
        print("\nMethod Analysis:")
        print("1. Raw Pixels:")
        print("   - Simple but limited representation")
        print("   - Sensitive to variations in pose, illumination")
        print("   - High dimensionality with limited semantic meaning")
        
        print("\n2. Local Binary Patterns (LBP):")
        print("   - Good texture descriptor")
        print("   - Rotation invariant to some degree")
        print("   - Compact representation")
        print("   - Works well for texture classification")
        
        print("\n3. Bag-of-Words (BoW):")
        print("   - Captures local patterns effectively")
        print("   - Translation invariant")
        print("   - Vocabulary size affects performance")
        print("   - Loses spatial information")
        
        print("\n4. Histogram of Oriented Gradients (HoG):")
        print("   - Excellent for capturing edge and shape information")
        print("   - Good balance of robustness and descriptiveness")
        print("   - Computationally efficient")
        print("   - Block normalization provides robustness")
        
        print(f"\nRecommendation: {methods[best_accuracy_idx]} provides the best balance")
        print("of accuracy and computational efficiency for this texture classification task.")
        
        print(f"\n📁 All visualizations and results saved to: {self.output_dir}/")
        print("Generated files:")
        for filename in sorted(os.listdir(self.output_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {filename}")
        
        return all_results

def main():
    """Main function to run the assignment solution."""
    # Set dataset path
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    
    # Create solution instance
    solution = CVAssignmentSolution(dataset_path)
    
    # Run complete analysis
    results = solution.run_complete_analysis()
    
    print("\n" + "="*50)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("All required components have been implemented:")
    print("✓ Integral image computation and verification")
    print("✓ Haar-like feature extraction (3 types)")
    print("✓ Raw pixel SVM classification")
    print("✓ LBP feature extraction and classification")
    print("✓ BoW feature extraction and classification")
    print("✓ HoG feature extraction and classification")
    print("✓ Performance comparison and analysis")

if __name__ == "__main__":
    main()