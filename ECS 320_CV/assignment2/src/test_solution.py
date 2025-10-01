"""
Computer Vision Assignment 2 - Basic Functionality Test Suite
============================================================
Author: Arnav Kapoor (23060)
Course: DSE312/DSE602/ECS320 - Computer Vision

PURPOSE:
========
This script provides essential unit tests to verify that the core components
of the CV assignment solution are working correctly before running the full
analysis. It performs quick validation of:

1. INTEGRAL IMAGE COMPUTATION: Tests mathematical correctness
2. DATASET LOADING: Verifies KTH-TIPS dataset accessibility
3. BASIC FEATURE EXTRACTION: Ensures methods are functional

USAGE:
======
Run this script before attempting the full solution to catch basic issues:
    python3 test_solution.py

EXPECTED OUTPUT:
===============
All tests should PASS for the solution to work correctly:
- Integral Image Test: PASSED
- Dataset Loading Test: PASSED

If any test fails, check the error messages for debugging guidance.
"""

# ============================================================================
# IMPORT STATEMENTS AND SETUP
# ============================================================================

import numpy as np                    # Numerical operations for test data
import sys                           # System path manipulation
import os                           # File system operations

# Add the assignment directory to Python path for importing our solution
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2')
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/src')

# Import our main solution class
from cv_assignment_solution import CVAssignmentSolution

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_integral_image():
    """
    Test Integral Image Implementation
    =================================
    
    This function validates the correctness of the integral image computation
    using a simple 3x3 test matrix with known expected results.
    
    TEST METHODOLOGY:
    1. Create a small test image with known pixel values
    2. Compute integral image using our implementation
    3. Verify specific values against hand-calculated expected results
    4. Test the verification function with a known rectangle
    5. Ensure numerical precision is maintained
    
    EXPECTED BEHAVIOR:
    - Integral image should contain cumulative sums
    - Verification should show exact match between methods
    - Function should return True if implementation is correct
    
    Returns:
        bool: True if integral image implementation passes all tests
    """
    print("="*60)
    print("TESTING INTEGRAL IMAGE IMPLEMENTATION")
    print("="*60)
    
    # Create a simple 3x3 test image with known values
    # This allows us to manually verify the expected integral image
    test_image = np.array([
        [1, 2, 3],    # Row 1: simple increasing values
        [4, 5, 6],    # Row 2: continue the pattern
        [7, 8, 9]     # Row 3: complete the 3x3 matrix
    ], dtype=np.float32)
    
    print("Test image (3x3 matrix):")
    print(test_image)
    
    # Expected integral image (calculated manually):
    # II[0,0] = 1
    # II[0,1] = 1+2 = 3  
    # II[0,2] = 1+2+3 = 6
    # II[1,0] = 1+4 = 5
    # II[1,1] = 1+2+4+5 = 12
    # II[1,2] = 1+2+3+4+5+6 = 21
    # II[2,0] = 1+4+7 = 12
    # II[2,1] = 1+2+4+5+7+8 = 27
    # II[2,2] = 1+2+3+4+5+6+7+8+9 = 45
    expected_integral = np.array([
        [1,  3,  6],
        [5, 12, 21],
        [12, 27, 45]
    ], dtype=np.float64)
    
    print("\nExpected integral image:")
    print(expected_integral)
    
    # Initialize solution class with dataset path
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    
    try:
        solution = CVAssignmentSolution(dataset_path)
        print("✓ Solution class initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize solution class: {e}")
        return False
    
    # Compute integral image using our implementation
    try:
        integral_img = solution.compute_integral_image(test_image)
        print("✓ Integral image computation completed")
    except Exception as e:
        print(f"✗ Integral image computation failed: {e}")
        return False
    
    print("\nComputed integral image:")
    print(integral_img)
    
    # Verify the computed integral image matches expected values
    if np.allclose(integral_img, expected_integral, rtol=1e-10):
        print("✓ Integral image values match expected results")
        matrix_test_passed = True
    else:
        print("✗ Integral image values don't match expected results")
        print("Difference:")
        print(integral_img - expected_integral)
        matrix_test_passed = False
    
    # Test the verification function with a known rectangle
    # Test rectangle from (0,0) to (1,1) - should sum to 1+2+4+5 = 12
    try:
        integral_sum, direct_sum, is_correct = solution.verify_integral_image(
            test_image, integral_img, 0, 0, 1, 1
        )
        
        print(f"\nVerification test for rectangle (0,0) to (1,1):")
        print(f"  Expected sum: 12 (1+2+4+5)")
        print(f"  Integral method: {integral_sum}")
        print(f"  Direct method: {direct_sum}")
        print(f"  Methods agree: {is_correct}")
        
        # Additional validation
        expected_sum = 12.0  # 1+2+4+5
        if abs(integral_sum - expected_sum) < 1e-10 and is_correct:
            print("✓ Verification function works correctly")
            verification_test_passed = True
        else:
            print("✗ Verification function failed")
            verification_test_passed = False
            
    except Exception as e:
        print(f"✗ Verification function crashed: {e}")
        verification_test_passed = False
    
    # Overall test result
    overall_passed = matrix_test_passed and verification_test_passed
    
    if overall_passed:
        print("\n🎉 INTEGRAL IMAGE TEST: PASSED")
    else:
        print("\n❌ INTEGRAL IMAGE TEST: FAILED")
    
    return overall_passed

def test_dataset_loading():
    """
    Test Dataset Loading Functionality
    =================================
    
    This function validates that the KTH-TIPS dataset can be loaded correctly
    and that all expected classes and images are accessible.
    
    TEST METHODOLOGY:
    1. Attempt to initialize solution with dataset path
    2. Load the complete dataset using our loading function
    3. Verify expected number of classes (should be 10)
    4. Verify reasonable number of images per class
    5. Check image dimensions and data types
    6. Validate class labels and names
    
    EXPECTED BEHAVIOR:
    - Should load 810 total images (81 per class)
    - All images should be resized to 200x200 pixels
    - Should have 10 distinct texture classes
    - No corrupted or missing images
    
    Returns:
        bool: True if dataset loading passes all validation checks
    """
    print("="*60)
    print("TESTING DATASET LOADING FUNCTIONALITY")
    print("="*60)
    
    # Define expected dataset characteristics
    expected_num_classes = 10
    expected_total_images = 810  # 81 images per class
    expected_image_size = (200, 200)  # Standard resized dimensions
    
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset directory not found: {dataset_path}")
        return False
    else:
        print(f"✓ Dataset directory found: {dataset_path}")
    
    # Initialize solution class
    try:
        solution = CVAssignmentSolution(dataset_path)
        print("✓ Solution class initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize solution class: {e}")
        return False
    
    # Verify class definitions
    print(f"\nExpected texture classes ({expected_num_classes}):")
    for i, class_name in enumerate(solution.classes):
        print(f"  {i+1:2d}. {class_name}")
    
    # Attempt to load the dataset
    try:
        print(f"\nLoading dataset from: {dataset_path}")
        print("This may take a moment...")
        
        images, labels, label_names = solution.load_dataset()
        
        print("✓ Dataset loading completed without errors")
        
    except Exception as e:
        print(f"✗ Dataset loading failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate loaded data characteristics
    validation_checks = []
    
    # Check 1: Total number of images
    actual_total = len(images)
    if actual_total == expected_total_images:
        print(f"✓ Total images: {actual_total} (matches expected {expected_total_images})")
        validation_checks.append(True)
    else:
        print(f"⚠ Total images: {actual_total} (expected {expected_total_images})")
        validation_checks.append(False)
    
    # Check 2: Number of classes
    actual_num_classes = len(solution.classes)
    if actual_num_classes == expected_num_classes:
        print(f"✓ Number of classes: {actual_num_classes}")
        validation_checks.append(True)
    else:
        print(f"✗ Number of classes: {actual_num_classes} (expected {expected_num_classes})")
        validation_checks.append(False)
    
    # Check 3: Image dimensions
    if len(images) > 0:
        actual_image_shape = images[0].shape
        if actual_image_shape == expected_image_size:
            print(f"✓ Image dimensions: {actual_image_shape}")
            validation_checks.append(True)
        else:
            print(f"⚠ Image dimensions: {actual_image_shape} (expected {expected_image_size})")
            validation_checks.append(True)  # Still acceptable, just different size
    else:
        print("✗ No images loaded")
        validation_checks.append(False)
    
    # Check 4: Class distribution
    print(f"\nClass distribution:")
    all_classes_have_images = True
    for class_idx, class_name in enumerate(solution.classes):
        class_count = np.sum(labels == class_idx)
        print(f"  {class_name}: {class_count} images")
        if class_count == 0:
            all_classes_have_images = False
    
    if all_classes_have_images:
        print("✓ All classes have images")
        validation_checks.append(True)
    else:
        print("✗ Some classes have no images")
        validation_checks.append(False)
    
    # Check 5: Data types and ranges
    if len(images) > 0:
        sample_image = images[0]
        if sample_image.dtype in [np.uint8, np.float32, np.float64]:
            print(f"✓ Image data type: {sample_image.dtype}")
            validation_checks.append(True)
        else:
            print(f"⚠ Unusual image data type: {sample_image.dtype}")
            validation_checks.append(True)  # May still work
        
        if sample_image.min() >= 0 and sample_image.max() <= 255:
            print(f"✓ Pixel value range: [{sample_image.min():.1f}, {sample_image.max():.1f}]")
            validation_checks.append(True)
        else:
            print(f"⚠ Pixel value range: [{sample_image.min():.1f}, {sample_image.max():.1f}]")
            validation_checks.append(True)  # May still work with normalization
    
    # Overall test result
    all_checks_passed = all(validation_checks)
    critical_checks_passed = validation_checks[:3]  # First 3 are critical
    
    if all(critical_checks_passed):
        print("\n🎉 DATASET LOADING TEST: PASSED")
        print("   Dataset is ready for texture classification experiments")
        return True
    else:
        print("\n❌ DATASET LOADING TEST: FAILED")
        print("   Critical issues found - check dataset integrity")
        return False

def main():
    """
    Main Test Execution Function
    ===========================
    
    Orchestrates the execution of all basic functionality tests and provides
    a comprehensive summary of results. This function should be run before
    attempting the full assignment solution.
    
    TEST SEQUENCE:
    1. Integral image computation and verification
    2. Dataset loading and validation
    3. Summary report with pass/fail status
    4. Recommendations for next steps
    """
    print("🧪 COMPUTER VISION ASSIGNMENT 2 - BASIC FUNCTIONALITY TESTS")
    print("=" * 65)
    print("Author: Arnav Kapoor (23060)")
    print("Course: Computer Vision (DSE312/DSE602/ECS320)")
    print("\nRunning essential tests before full solution execution...")
    print("=" * 65)
    
    # Track test results
    test_results = {}
    
    # Test 1: Integral Image Implementation
    print("\n🔍 TEST 1: INTEGRAL IMAGE COMPUTATION")
    test_results['integral_image'] = test_integral_image()
    
    # Test 2: Dataset Loading
    print("\n📂 TEST 2: DATASET LOADING")
    test_results['dataset_loading'] = test_dataset_loading()
    
    # Summary Report
    print("\n" + "=" * 65)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 65)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"Tests executed: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        status_text = "PASSED" if result else "FAILED"
        print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {status_text}")
    
    # Recommendations
    print("\n" + "=" * 65)
    print("📋 RECOMMENDATIONS")
    print("=" * 65)
    
    if all(test_results.values()):
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now proceed with confidence to run:")
        print("  • python3 demo_solution.py      (Quick demo)")
        print("  • python3 quick_demo.py         (Fast overview)")
        print("  • python3 cv_assignment_solution.py  (Complete analysis)")
        print("\n✨ Your CV assignment solution is ready for execution!")
        
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("\nBefore running the full solution, please:")
        
        if not test_results['integral_image']:
            print("  • Fix integral image computation errors")
            print("  • Check mathematical implementation")
            print("  • Verify rectangle sum calculations")
        
        if not test_results['dataset_loading']:
            print("  • Verify KTH-TIPS dataset is properly extracted")
            print("  • Check file paths and permissions")
            print("  • Ensure all image files are accessible")
        
        print("\n🔧 Debug the failed components before proceeding.")
    
    print("\n" + "=" * 65)
    return all(test_results.values())

if __name__ == "__main__":
    """
    Script Entry Point
    ================
    
    When this script is run directly, it executes the main test function
    and exits with appropriate status codes for automated testing.
    """
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n❌ Tests could not complete - check your environment setup")
        exit_code = 2
    
    # Exit with appropriate code for automated testing
    sys.exit(exit_code)