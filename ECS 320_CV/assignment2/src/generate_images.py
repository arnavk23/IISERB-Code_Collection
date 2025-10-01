"""
Image Generation Script for CV Assignment
This script generates and saves all visualizations without running full analysis
"""

import sys
import os
sys.path.append('/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2')

from cv_assignment_solution import CVAssignmentSolution
import numpy as np
import matplotlib.pyplot as plt

def generate_all_images():
    """Generate and save all possible images and visualizations"""
    print("=" * 60)
    print("CV ASSIGNMENT - IMAGE GENERATION")
    print("Generating all visualizations and saving to disk...")
    print("=" * 60)
    
    dataset_path = "/home/arnav/Downloads/23060_Arnav_Kapoor/assignment2/kth_tips_col_200x200/KTH_TIPS"
    solution = CVAssignmentSolution(dataset_path)
    
    print(f"Output directory: {solution.output_dir}")
    
    # 1. Generate Integral Image and Haar Feature Demo
    print("\n🔍 Generating Integral Image and Haar Feature visualizations...")
    
    test_image = solution.create_cameraman_image()
    integral_img = solution.compute_integral_image(test_image)
    
    # Save test image
    plt.figure(figsize=(8, 6))
    plt.imshow(test_image, cmap='gray')
    plt.title('Test Image for Haar Feature Extraction')
    plt.colorbar()
    filename = f"{solution.output_dir}/test_image.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # Save integral image
    plt.figure(figsize=(8, 6))
    plt.imshow(integral_img, cmap='viridis')
    plt.title('Integral Image Representation')
    plt.colorbar()
    filename = f"{solution.output_dir}/integral_image.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # Extract and visualize Haar features
    center_x, center_y = (256 - 50) // 2, (256 - 50) // 2
    haar_features = solution.extract_haar_features(test_image, center_x, center_y, 50, 24)
    
    # Create Haar region visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(test_image, cmap='gray')
    from matplotlib.patches import Rectangle
    rect = Rectangle((center_y, center_x), 50, 50, 
                   linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title('Haar Feature Extraction Region (50×50)')
    plt.colorbar()
    filename = f"{solution.output_dir}/haar_extraction_region.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # 2. Load dataset and save samples
    print("\n📂 Loading dataset and saving sample images...")
    
    images, labels, _ = solution.load_dataset()
    
    # Save more sample images per class
    solution.save_sample_images(images, labels, n_samples=5)
    
    # Save feature extraction examples
    solution.save_feature_examples(images, labels)
    
    # 3. Generate feature comparison visualizations
    print("\n🔧 Generating feature extraction comparisons...")
    
    # Use subset for faster processing
    n_samples = 20  # 2 per class for quick demo
    demo_images = []
    demo_labels = []
    
    for class_idx in range(len(solution.classes)):
        class_mask = labels == class_idx
        class_images = images[class_mask][:2]
        class_labels = labels[class_mask][:2]
        demo_images.extend(class_images)
        demo_labels.extend(class_labels)
    
    demo_images = np.array(demo_images)
    demo_labels = np.array(demo_labels)
    
    # Generate LBP examples for multiple images
    print("  Generating LBP examples...")
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for i in range(3):
        img = demo_images[i]
        lbp_img = solution.compute_lbp(img)
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original: {solution.classes[demo_labels[i]]}')
        axes[i, 0].axis('off')
        
        # LBP image
        axes[i, 1].imshow(lbp_img, cmap='gray')
        axes[i, 1].set_title('LBP Pattern')
        axes[i, 1].axis('off')
        
        # LBP histogram
        hist, bins = np.histogram(lbp_img.flatten(), bins=256, range=(0, 256))
        axes[i, 2].bar(range(len(hist)), hist, width=1)
        axes[i, 2].set_title('LBP Histogram')
        axes[i, 2].set_xlabel('LBP Value')
        axes[i, 2].set_ylabel('Frequency')
        
        # HoG visualization
        hog_img, magnitude, direction = solution.visualize_hog(img)
        
        axes[i, 3].imshow(magnitude, cmap='hot')
        axes[i, 3].set_title('Gradient Magnitude')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(direction, cmap='hsv')
        axes[i, 4].set_title('Gradient Direction')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(hog_img, cmap='gray')
        axes[i, 5].set_title('HoG Visualization')
        axes[i, 5].axis('off')
    
    plt.suptitle('Feature Extraction Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = f"{solution.output_dir}/feature_methods_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # 4. Generate class distribution visualization
    print("  Generating dataset overview...")
    
    class_counts = [np.sum(labels == i) for i in range(len(solution.classes))]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(solution.classes, class_counts, color=plt.cm.tab10(range(len(solution.classes))))
    plt.title('KTH-TIPS Dataset - Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Texture Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"{solution.output_dir}/dataset_distribution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # 5. Create a collage of sample images from all classes
    print("  Creating class samples collage...")
    
    plt.figure(figsize=(20, 12))
    samples_per_class = 4
    
    for class_idx, class_name in enumerate(solution.classes):
        class_mask = labels == class_idx
        class_images = images[class_mask][:samples_per_class]
        
        for sample_idx, img in enumerate(class_images):
            plt.subplot(len(solution.classes), samples_per_class, 
                       class_idx * samples_per_class + sample_idx + 1)
            plt.imshow(img, cmap='gray')
            if sample_idx == 0:
                plt.ylabel(class_name, fontsize=12, fontweight='bold')
            plt.title(f'Sample {sample_idx + 1}')
            plt.axis('off')
    
    plt.suptitle('KTH-TIPS Dataset - Sample Images from All Classes', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = f"{solution.output_dir}/all_classes_samples.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # 6. Generate algorithm flowchart visualization
    print("  Creating algorithm overview...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Integral Image Process
    ax1.text(0.5, 0.9, 'INTEGRAL IMAGE COMPUTATION', ha='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.7, '1. Input: Grayscale Image I(x,y)', ha='center', fontsize=11)
    ax1.text(0.5, 0.6, '2. Compute: II(x,y) = Σ I(i,j) for i≤x, j≤y', ha='center', fontsize=11)
    ax1.text(0.5, 0.5, '3. Rectangle Sum: O(1) lookup', ha='center', fontsize=11)
    ax1.text(0.5, 0.4, '4. Apply Haar Features', ha='center', fontsize=11)
    ax1.text(0.5, 0.2, 'Time Complexity: O(n²) → O(1) queries', ha='center', fontsize=10, style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Feature Extraction Methods
    ax2.text(0.5, 0.9, 'FEATURE EXTRACTION METHODS', ha='center', fontsize=14, fontweight='bold')
    methods_text = """
Raw Pixels: Direct pixel intensities
LBP: Local Binary Pattern histograms  
BoW: Bag-of-Words from patches
HoG: Histogram of Oriented Gradients

Each method captures different aspects:
• Raw: Pixel-level information
• LBP: Local texture patterns
• BoW: Repeated visual patterns  
• HoG: Edge and shape information
"""
    ax2.text(0.05, 0.8, methods_text, fontsize=10, verticalalignment='top')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Classification Pipeline
    ax3.text(0.5, 0.9, 'CLASSIFICATION PIPELINE', ha='center', fontsize=14, fontweight='bold')
    pipeline_text = """
1. Dataset Loading (810 images, 10 classes)
2. Feature Extraction (4 methods)
3. Train-Test Split (70-30)
4. Feature Standardization
5. SVM Training (RBF kernel)
6. Performance Evaluation
7. Results Comparison

Metrics: Accuracy, Confusion Matrix,
Training Time, Prediction Time
"""
    ax3.text(0.05, 0.8, pipeline_text, fontsize=10, verticalalignment='top')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Haar Feature Types
    ax4.text(0.5, 0.9, 'HAAR FEATURE TYPES', ha='center', fontsize=14, fontweight='bold')
    haar_text = """
Type 1: Left-Right Pattern
[Light | Dark] - Detects vertical edges

Type 2: Top-Bottom Pattern  
[Light]
[Dark ] - Detects horizontal edges

Type 3: Three-Rectangle Pattern
[Light|Dark|Light] - Detects lines

Applications: Object detection, face
recognition, texture analysis
"""
    ax4.text(0.05, 0.8, haar_text, fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    filename = f"{solution.output_dir}/algorithm_overview.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ IMAGE GENERATION COMPLETED!")
    print("=" * 60)
    
    # List all generated files
    saved_files = []
    if os.path.exists(solution.output_dir):
        for filename in os.listdir(solution.output_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                saved_files.append(filename)
    
    print(f"📁 Total images generated: {len(saved_files)}")
    print(f"📂 Location: {solution.output_dir}/")
    print("\nGenerated files:")
    
    # Group files by category
    categories = {
        'Dataset Samples': [f for f in saved_files if 'sample_' in f],
        'Feature Visualizations': [f for f in saved_files if any(x in f for x in ['lbp', 'hog', 'feature'])],
        'Integral & Haar': [f for f in saved_files if any(x in f for x in ['integral', 'haar', 'test'])],
        'Analysis & Overview': [f for f in saved_files if any(x in f for x in ['distribution', 'overview', 'algorithm', 'comparison', 'collage', 'all_classes'])]
    }
    
    for category, files in categories.items():
        if files:
            print(f"\n{category}:")
            for filename in sorted(files):
                print(f"  • {filename}")
    
    return len(saved_files)

if __name__ == "__main__":
    try:
        num_images = generate_all_images()
        print(f"\n🎉 Successfully generated {num_images} images!")
    except Exception as e:
        print(f"\n❌ Image generation failed: {e}")
        import traceback
        traceback.print_exc()