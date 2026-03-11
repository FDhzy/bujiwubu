"""
Test shoe recognition stability - verify consistent results for same object.
"""
import cv2
import numpy as np
from models.shoe_analyzer import ShoeAnalyzer, ShoeType

def create_test_shoe_image():
    """Create a consistent test shoe image."""
    # Create a simple shoe-like shape
    img = np.ones((400, 300, 3), dtype=np.uint8) * 220  # Light background
    
    # Draw a shoe shape (low shoe / sneaker)
    # Foot part
    cv2.rectangle(img, (60, 200), (240, 350), (80, 100, 120), -1)
    cv2.ellipse(img, (150, 200), (90, 60), 0, 0, 180, (80, 100, 120), -1)
    
    # Sole
    cv2.rectangle(img, (60, 330), (240, 350), (50, 60, 70), -1)
    
    # Add some texture
    for i in range(5):
        y = 220 + i * 20
        cv2.line(img, (80, y), (220, y), (70, 90, 110), 1)
    
    return img


def add_noise_variation(img, intensity=10):
    """Add slight noise/lighting variation to image."""
    noise = np.random.randint(-intensity, intensity, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def test_consistency():
    """Test that same shoe gets same recognition consistently."""
    print("\n" + "="*60)
    print("SHOE RECOGNITION CONSISTENCY TEST")
    print("="*60)
    print("\nTesting: Taking 10 photos of the SAME shoe object")
    print("Expected: Should get the same shoe type every time")
    print("-"*60)
    
    # Create analyzer
    analyzer = ShoeAnalyzer(language="en")
    
    # Create base test image
    base_image = create_test_shoe_image()
    
    # Test multiple times with slight variations (simulating real captures)
    results = []
    shoe_types_detected = []
    
    for i in range(10):
        # Add slight noise to simulate real camera variation
        test_image = add_noise_variation(base_image, intensity=8)
        
        # Analyze
        result = analyzer.analyze(test_image)
        results.append(result)
        shoe_types_detected.append(result.shoe_type.value)
        
        print(f"Capture {i+1:2d}: {result.shoe_type.value:15} (confidence: {result.confidence:.2%})")
    
    # Check consistency
    print("-"*60)
    unique_types = set(shoe_types_detected)
    print(f"\nUnique shoe types detected: {len(unique_types)}")
    print(f"Types: {unique_types}")
    
    # Count occurrences
    from collections import Counter
    counts = Counter(shoe_types_detected)
    print(f"\nType frequency:")
    for shoe_type, count in counts.most_common():
        percentage = (count / len(shoe_types_detected)) * 100
        print(f"  {shoe_type:15}: {count:2d} times ({percentage:.1f}%)")
    
    # Determine result
    print("\n" + "="*60)
    if len(unique_types) == 1:
        print("✓ PERFECT CONSISTENCY - Same type every time!")
        consistency_score = 100
    else:
        most_common = counts.most_common(1)[0]
        consistency_score = (most_common[1] / len(shoe_types_detected)) * 100
        print(f"Consistency Score: {consistency_score:.1f}%")
        
        if consistency_score >= 80:
            print("✓ GOOD CONSISTENCY - Stable recognition")
        elif consistency_score >= 60:
            print("⚠ MODERATE CONSISTENCY - Some variation")
        else:
            print("✗ POOR CONSISTENCY - Unstable recognition")
    
    print("="*60)
    
    return consistency_score >= 80


def test_different_shoes():
    """Test that DIFFERENT shoes get different recognition."""
    print("\n" + "="*60)
    print("DIFFERENT SHOE DISCRIMINATION TEST")
    print("="*60)
    print("\nTesting: Analyzing 3 different shoe types")
    print("Expected: Should correctly identify each as different")
    print("-"*60)
    
    analyzer = ShoeAnalyzer(language="en")
    
    # Test with 3 different shoe images
    test_cases = [
        ("Sandal (flat)", (400, 250)),
        ("Low Shoe", (400, 300)),
        ("Boot (tall)", (500, 300))
    ]
    
    detected_types = []
    
    for name, size in test_cases:
        # Create image
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 220
        
        # Draw shape based on type
        h, w = size
        if "Sandal" in name:
            # Flat sandal
            cv2.rectangle(img, (50, h-80), (w-50, h-50), (150, 130, 100), -1)
        elif "Low" in name:
            # Low shoe
            cv2.rectangle(img, (60, h-150), (w-60, h-50), (80, 100, 120), -1)
        else:
            # Tall boot
            cv2.rectangle(img, (80, h-350), (w-80, h-50), (60, 70, 90), -1)
        
        # Analyze multiple times for stability
        types = []
        for _ in range(3):
            test_img = add_noise_variation(img, intensity=5)
            result = analyzer.analyze(test_img)
            types.append(result.shoe_type.value)
        
        # Use most common
        from collections import Counter
        most_common = Counter(types).most_common(1)[0][0]
        detected_types.append(most_common)
        
        print(f"{name:20} -> Detected as: {most_common}")
        analyzer.reset_detection_cache()  # Reset for next shoe
    
    print("-"*60)
    unique_detected = len(set(detected_types))
    print(f"\nUnique types detected: {unique_detected} out of 3")
    
    print("\n" + "="*60)
    if unique_detected == 3:
        print("✓ EXCELLENT - All shoes detected as different types")
        return True
    elif unique_detected == 2:
        print("⚠ PARTIAL - Some shoes confused")
        return False
    else:
        print("✗ POOR - All shoes detected as same type")
        return False


def main():
    """Run all stability tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*12 + "SHOE RECOGNITION STABILITY TEST" + " "*15 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\nThis test verifies the bug fix for inconsistent shoe recognition.")
    print("The issue: Same shoe object getting different types on repeated photos.")
    print("The fix: Added preprocessing, hysteresis, and majority voting.")
    
    # Run tests
    test1_passed = test_consistency()
    test2_passed = test_different_shoes()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Consistency Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Discrimination Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED - Bug is fixed!")
        print("The shoe recognition is now stable and consistent.")
        return 0
    else:
        print("\n⚠ SOME TESTS FAILED - Further tuning may be needed")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
