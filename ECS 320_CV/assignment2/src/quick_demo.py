"""
Computer Vision Assignment 2 - Quick Demonstration Script
========================================================
Author: Arnav Kapoor (23060)
Roll Number: 23060
Course: DSE312/DSE602/ECS320 - Computer Vision

PURPOSE:
========
This script provides a fast demonstration of the key components of the
CV assignment solution. It's designed to quickly verify that all major
functionalities are working without running the complete time-intensive
analysis.

DEMONSTRATION INCLUDES:
======================
1. INTEGRAL IMAGE COMPUTATION
   - Creates test image and computes integral representation
   - Verifies correctness with mathematical validation
   - Extracts Haar-like features from test region

2. DATASET PROCESSING  
   - Loads KTH-TIPS texture dataset
   - Demonstrates feature extraction methods
   - Shows classification pipeline with small subset

3. FEATURE EXTRACTION SHOWCASE
   - Raw pixel feature extraction
   - Local Binary Pattern (LBP) computation
   - Quick SVM classification demonstration

EXECUTION TIME:
===============
This demo runs in approximately 30-60 seconds, making it ideal for:
- Quick functionality verification
- Debugging and development
- Demonstrating solution capabilities
- Pre-submission validation

USAGE:
======
Simply run: python3 quick_demo.py

The script will output progress indicators and final success confirmation.
"""

# ============================================================================
# IMPORT STATEMENTS AND ENVIRONMENT SETUP
# ============================================================================

import sys                          # System interface for path manipulation
import os                           # Operating system interface for file checks
import numpy as np                  # Numerical operations for data handling

# Add project directories to Python path for imports
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2')
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/src')

# Import our comprehensive CV assignment solution
from cv_assignment_solution import CVAssignmentSolution

# ============================================================================
# MAIN DEMONSTRATION FUNCTION
# ============================================================================

def main():
    """
    Quick Demonstration of CV Assignment Solution
    ===========================================
    
    This function orchestrates a rapid demonstration of all key components
    of the Computer Vision assignment solution, providing confidence that
    the implementation is working correctly.
    
    DEMONSTRATION WORKFLOW:
    1. Environment validation and setup
    2. Integral image and Haar feature demonstration  
    3. Dataset loading and preprocessing showcase
    4. Feature extraction method examples
    5. Classification pipeline validation
    6. Success confirmation and next steps
    
    Returns:
        bool: True if all demonstrations complete successfully
    """
    # Display header information
    print("=" * 60)
    print("COMPUTER VISION ASSIGNMENT 2 - QUICK DEMO")
    print("=" * 60)
    print("Student: Arnav Kapoor (23060)")
    print("Course: DSE312/DSE602/ECS320 - Computer Vision")
    print("Institution: IISER Bhopal")
    print("Date: September 2025")
    print("=" * 60)
    
    # Define dataset path (adjust if needed for different environments)
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    
    # STEP 1: ENVIRONMENT VALIDATION
    print("\nSTEP 1: ENVIRONMENT VALIDATION")
    print("-" * 40)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        print(f"Expected location: {dataset_path}")
        print("Please ensure KTH-TIPS dataset is extracted correctly.")
        return False
    else:
        print("Dataset directory found")
        print(f"Location: {dataset_path}")
    
    # Initialize solution class
    try:
        solution = CVAssignmentSolution(dataset_path)
        print("Solution class initialized successfully")
        print(f"Output directory: {solution.output_dir}")
    except Exception as e:
        print(f"Failed to initialize solution: {e}")
        return False
    
    # STEP 2: INTEGRAL IMAGE AND HAAR FEATURES DEMO
    print("\nSTEP 2: INTEGRAL IMAGE AND HAAR FEATURES")
    print("-" * 50)
    
    try:
        # Create a synthetic test image for Haar feature demonstration
        # This simulates the "cameraman" image commonly used in computer vision
        print("Creating synthetic test image...")
        test_image = solution.create_cameraman_image()
        print(f"Test image created: {test_image.shape} pixels")
        print(f"Pixel value range: [{test_image.min():.1f}, {test_image.max():.1f}]")

        # Compute integral image representation
        # This is the core data structure for fast Haar feature computation
        print("\nComputing integral image...")
        integral_img = solution.compute_integral_image(test_image)
        print(f"Integral image computed: {integral_img.shape}")
        print(f"Max integral value: {integral_img.max():.0f}")

        # Verify implementation correctness using mathematical validation
        print("\nVerifying integral image correctness...")
        test_x1, test_y1, test_x2, test_y2 = 10, 10, 50, 50  # Test rectangle
        integral_sum, direct_sum, is_correct = solution.verify_integral_image(
            test_image, integral_img, test_x1, test_y1, test_x2, test_y2
        )
        
        if is_correct:
            print(f"Verification PASSED")
            print(f"Rectangle ({test_x1},{test_y1}) to ({test_x2},{test_y2})")
            print(f"Integral method: {integral_sum:.2f}")
            print(f"Direct method: {direct_sum:.2f}")
            print(f"Difference: {abs(integral_sum - direct_sum):.2e}")
        else:
            print(f"Verification FAILED - check implementation")
            return False
        
        # Extract Haar-like features from center region (as required by assignment)
        print("\nExtracting Haar-like features...")
        center_x, center_y = 100, 100  # Center region coordinates
        region_size = 50                # 50x50 region as specified
        min_filter_size = 24            # Minimum filter size as required
        
        haar_features = solution.extract_haar_features(
            test_image, center_x, center_y, region_size, min_filter_size
        )
        
        # Report feature extraction results
        type1_count = len(haar_features['type1'])  # Left-Right patterns
        type2_count = len(haar_features['type2'])  # Top-Bottom patterns  
        type3_count = len(haar_features['type3'])  # Three-rectangle patterns
        total_features = type1_count + type2_count + type3_count

        print(f"Haar feature extraction completed:")
        print(f"Type 1 (Left-Right): {type1_count} features")
        print(f"Type 2 (Top-Bottom): {type2_count} features")
        print(f"Type 3 (Three-rect): {type3_count} features")
        print(f"Total features: {total_features}")

    except Exception as e:
        print(f"Integral image/Haar features demo failed: {e}")
        return False
    
    # STEP 3: TEXTURE CLASSIFICATION DEMO
    print("\nSTEP 3: TEXTURE CLASSIFICATION DEMO")
    print("-" * 50)
    
    try:
        # Load the complete KTH-TIPS dataset
        print("Loading KTH-TIPS texture dataset...")
        print("This may take a moment for initial loading...")
        
        images, labels, label_names = solution.load_dataset()

        print(f"Dataset loaded successfully:")
        print(f"Total images: {len(images)}")
        print(f"Number of classes: {len(solution.classes)}")
        print(f"Image dimensions: {images[0].shape}")
        print(f"Classes: {', '.join(solution.classes[:5])}...")

        # Create a small subset for quick demonstration
        # Using fewer images to keep demo fast while showing functionality
        print("\nCreating demo subset...")
        samples_per_class = 5  # Use 5 images per class for speed
        demo_images = []
        demo_labels = []
        
        for class_idx in range(len(solution.classes)):
            # Get images belonging to current class
            class_mask = labels == class_idx
            class_images = images[class_mask][:samples_per_class]
            class_labels = labels[class_mask][:samples_per_class]
            
            # Add to demo dataset
            demo_images.extend(class_images)
            demo_labels.extend(class_labels)
        
        # Convert to numpy arrays for processing
        demo_images = np.array(demo_images)
        demo_labels = np.array(demo_labels)

        print(f"Demo subset created:")
        print(f"Demo images: {len(demo_images)} ({samples_per_class} per class)")
        print(f"Demo shape: {demo_images.shape}")

    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return False
    
    # STEP 4: FEATURE EXTRACTION SHOWCASE
    print("\nSTEP 4: FEATURE EXTRACTION METHODS")
    print("-" * 50)
    
    try:
        # Demonstrate Raw Pixel Feature Extraction
        print("Testing Raw Pixel Features...")
        target_size = (32, 32)  # Smaller size for quick demo
        raw_features = solution.extract_raw_pixel_features(demo_images, target_size)
        print(f"Raw pixels: {raw_features.shape}")
        print(f"Feature vector size: {raw_features.shape[1]} pixels")
        print(f"Memory usage: {raw_features.nbytes / 1024:.1f} KB")

        # Demonstrate Local Binary Pattern Feature Extraction
        print("\n Testing Local Binary Pattern (LBP) Features...")
        lbp_features = solution.extract_lbp_features(demo_images)
        print(f"LBP features: {lbp_features.shape}")
        print(f"Histogram bins: {lbp_features.shape[1]}")
        print(f"Texture patterns captured: {np.count_nonzero(lbp_features[0])}")

        # Report feature extraction success
        print(f"\nFeature extraction methods validated")
        print(f"Both methods completed without errors")
        print(f"Ready for classification pipeline")

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return False
    
    # STEP 5: CLASSIFICATION PIPELINE VALIDATION
    print("\nSTEP 5: CLASSIFICATION PIPELINE VALIDATION")
    print("-" * 50)
    
    try:
        # Perform quick classification test using raw pixel features
        print("Testing SVM classification pipeline...")
        print("Using raw pixel features for quick demonstration...")

        # Train and evaluate classifier (this will split data internally)
        results = solution.train_and_evaluate_classifier(
            raw_features, demo_labels, "Quick Demo Classification"
        )
        
        # Extract and report key metrics
        accuracy = results['accuracy']
        training_time = results['training_time']
        prediction_time = results['prediction_time']
        
        print(f"Classification completed:")
        print(f"Accuracy: {accuracy:.1%} ({accuracy:.4f})")
        print(f"Training time: {training_time:.3f} seconds")
        print(f"Prediction time: {prediction_time:.4f} seconds")

        # Interpret results for user
        if accuracy > 0.3:  # Reasonable for small subset and quick demo
            print(f"Performance: Acceptable for demo subset")
        else:
            print(f"Performance: Low (expected for small subset)")

        print(f"Note: Full dataset will yield significantly better results")

    except Exception as e:
        print(f"Classification pipeline failed: {e}")
        return False
    
    # STEP 6: SUCCESS CONFIRMATION AND NEXT STEPS
    print("\n" + "=" * 60)
    print("QUICK DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Summary of what was demonstrated
    demo_components = [
        "✓ Integral image computation and verification",
        "✓ Haar-like feature extraction (3 types)", 
        "✓ Dataset loading and preprocessing",
        "✓ Multiple feature extraction methods",
        "✓ SVM classification pipeline",
        "✓ Performance evaluation metrics"
    ]
    
    print("All major components are working correctly:")
    for component in demo_components:
        print(f"   {component}")
    
    # Provide guidance for next steps
    print("\nNEXT STEPS:")
    print("For complete analysis with all 810 images:")
    print("python3 cv_assignment_solution.py")
    print("")
    print("For extended demo with more features:")
    print("python3 demo_solution.py")
    print("")
    print("For generating all visualizations:")
    print("python3 generate_images.py")

    # Check if images were generated during demo
    if hasattr(solution, 'output_dir') and os.path.exists(solution.output_dir):
        generated_files = [f for f in os.listdir(solution.output_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        if generated_files:
            print(f"\nGenerated visualizations in {solution.output_dir}/:")
            for filename in sorted(generated_files[:5]):  # Show first 5
                print(f" {filename}")
            if len(generated_files) > 5:
                print(f" ... and {len(generated_files) - 5} more files")

    print("\nYour Computer Vision assignment solution is ready!")
    return True

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Script Entry Point
    ================
    
    When this script is run directly, it executes the quick demonstration
    and provides appropriate exit codes for automated testing and integration.
    
    EXIT CODES:
    - 0: Demo completed successfully, all components working
    - 1: Demo failed due to functional issues
    - 2: Demo failed due to critical environment/setup errors
    """
    try:
        print("Starting Computer Vision Assignment Quick Demo...")
        print("Estimated runtime: 30-60 seconds")
        
        # Execute the main demonstration
        success = main()
        
        if success:
            print("\nDemo completed successfully!")
            print("All major components verified and working")
            exit_code = 0
        else:
            print("\nDemo failed!")
            print("Please check error messages above for debugging")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user (Ctrl+C)")
        print("You can restart the demo at any time")
        exit_code = 1
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Full error traceback:")
        import traceback
        traceback.print_exc()
        print("\nPlease check your environment setup:")
        print("• Python dependencies installed")
        print("• Dataset extracted correctly") 
        print("• File permissions accessible")
        exit_code = 2
    
    # Exit with appropriate status code
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)