"""
Demo script for CV Assignment - Runs a smaller version for testing
"""

import sys
import os
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2')

from cv_assignment_solution import CVAssignmentSolution
import numpy as np
import matplotlib.pyplot as plt

def run_demo():
    """Run a smaller demo of the assignment"""
    print("=== CV Assignment Demo ===")
    
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    solution = CVAssignmentSolution(dataset_path)
    
    print(f"Output directory created: {solution.output_dir}")
    
    # Part 1: Integral Image and Haar Features Demo
    print("\n1. Testing Integral Image and Haar Features...")
    
    # Create test image
    test_image = solution.create_cameraman_image()
    print(f"Created test image of size: {test_image.shape}")
    
    # Test integral image
    integral_img = solution.compute_integral_image(test_image)
    print(f"Computed integral image of size: {integral_img.shape}")
    
    # Verify correctness
    x1, y1, x2, y2 = 10, 10, 50, 50
    integral_sum, direct_sum, is_correct = solution.verify_integral_image(
        test_image, integral_img, x1, y1, x2, y2
    )
    print(f"Verification test - Correct: {is_correct}")
    
    # Extract Haar features
    center_x, center_y = (256 - 50) // 2, (256 - 50) // 2
    haar_features = solution.extract_haar_features(
        test_image, center_x, center_y, 50, min_filter_size=24
    )
    print(f"Extracted Haar features:")
    print(f"  Type 1: {len(haar_features['type1'])} features")
    print(f"  Type 2: {len(haar_features['type2'])} features") 
    print(f"  Type 3: {len(haar_features['type3'])} features")
    
    # Part 2: Texture Classification Demo (using subset of data)
    print("\n2. Testing Texture Classification (using subset)...")
    
    # Load dataset
    images, labels, label_names = solution.load_dataset()
    print(f"Loaded {len(images)} images from {len(solution.classes)} classes")
    
    # Use only first 100 images for demo (10 per class)
    n_samples_per_class = 10
    demo_images = []
    demo_labels = []
    
    for class_idx in range(len(solution.classes)):
        class_mask = labels == class_idx
        class_images = images[class_mask][:n_samples_per_class]
        class_labels = labels[class_mask][:n_samples_per_class]
        
        demo_images.extend(class_images)
        demo_labels.extend(class_labels)
    
    demo_images = np.array(demo_images)
    demo_labels = np.array(demo_labels)
    
    print(f"Using {len(demo_images)} images for demo ({n_samples_per_class} per class)")
    
    # Test Raw Pixel Classification
    print("\n  Testing Raw Pixel Features...")
    raw_features = solution.extract_raw_pixel_features(demo_images, target_size=(32, 32))
    print(f"  Raw feature shape: {raw_features.shape}")
    
    # Test LBP Features
    print("\n  Testing LBP Features...")
    lbp_features = solution.extract_lbp_features(demo_images)
    print(f"  LBP feature shape: {lbp_features.shape}")
    
    # Test one classification (Raw Pixel)
    print("\n  Training Raw Pixel Classifier...")
    raw_results = solution.train_and_evaluate_classifier(raw_features, demo_labels, "Raw Pixels (Demo)")
    
    print("\n=== Demo Results ===")
    print(f"Raw Pixel Classification Accuracy: {raw_results['accuracy']:.4f}")
    print(f"Training time: {raw_results['training_time']:.2f} seconds")
    
    print("\n=== Demo Completed Successfully! ===")
    print("All major components are working correctly.")
    print("You can now run the full solution using cv_assignment_solution.py")
    
    # List saved files
    if os.path.exists(solution.output_dir):
        saved_files = [f for f in os.listdir(solution.output_dir) if f.endswith(('.png', '.jpg'))]
        if saved_files:
            print(f"\n📁 Images saved to {solution.output_dir}/:")
            for filename in sorted(saved_files):
                print(f"  - {filename}")
    
    return True

if __name__ == "__main__":
    try:
        success = run_demo()
        if success:
            print("\n✓ Demo completed successfully!")
        else:
            print("\n✗ Demo failed!")
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()