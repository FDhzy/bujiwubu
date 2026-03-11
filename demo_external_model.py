"""
Simple demonstration of external model integration.
This script shows the complete workflow.
"""
import os
import cv2
import numpy as np
from models.shoe_analyzer import (
    ShoeAnalyzer,
    ShoeType,
    TensorFlowShoeModel,
    CustomModelAdapter
)


def demo_without_external_model():
    """Demo using built-in CV method only."""
    print("\n" + "="*60)
    print("DEMO 1: Built-in Computer Vision Method")
    print("="*60)
    
    # Create analyzer without external model
    analyzer = ShoeAnalyzer(language="en")
    
    # Create a test image (simulated shoe)
    test_image = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 100), (250, 350), (100, 100, 150), -1)
    
    # Analyze
    result = analyzer.analyze(test_image)
    
    print(f"Detected shoe type: {result.shoe_type.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Safe puddle depth: {result.safe_puddle_depth_cm:.1f} cm")
    print(f"Description: {result.description}")


def demo_with_custom_function():
    """Demo using a custom prediction function."""
    print("\n" + "="*60)
    print("DEMO 2: Custom Prediction Function")
    print("="*60)
    
    # Define a simple custom predictor
    def my_smart_predictor(image: np.ndarray):
        """
        This could be any ML model you want.
        For demo, we'll use a simple rule.
        """
        h, w = image.shape[:2]
        aspect = h / w
        
        # Simulate better prediction than CV
        if aspect > 1.5:
            return ShoeType.HIGH_BOOTS, 0.92
        elif aspect > 1.0:
            return ShoeType.BOOTS, 0.88
        elif aspect > 0.7:
            return ShoeType.ANKLE_SHOES, 0.85
        else:
            return ShoeType.LOW_SHOES, 0.82
    
    # Wrap it in adapter
    custom_model = CustomModelAdapter(my_smart_predictor)
    
    # Create analyzer with custom model
    analyzer = ShoeAnalyzer(language="en", external_model=custom_model)
    
    # Test with tall image (rain boot)
    test_image = np.zeros((500, 250, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (200, 450), (80, 80, 120), -1)
    
    # Analyze
    result = analyzer.analyze(test_image)
    
    print(f"Detected shoe type: {result.shoe_type.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Safe puddle depth: {result.safe_puddle_depth_cm:.1f} cm")
    print(f"Description: {result.description}")
    print("\nNote: Higher confidence because we used 'external model'")


def demo_with_tensorflow_model():
    """Demo attempting to use TensorFlow model."""
    print("\n" + "="*60)
    print("DEMO 3: TensorFlow Lite Model (if available)")
    print("="*60)
    
    # Try to create TF model
    tf_model = TensorFlowShoeModel()
    
    # Try to load a model (this will fail if no model exists)
    model_path = "assets/models/shoe_classifier.tflite"
    
    if os.path.exists(model_path):
        success = tf_model.load_model(model_path)
        if success:
            analyzer = ShoeAnalyzer(language="en", external_model=tf_model)
            print(f"✓ Successfully loaded model: {model_path}")
            
            # Test it
            test_image = np.zeros((400, 300, 3), dtype=np.uint8)
            result = analyzer.analyze(test_image)
            print(f"Detected: {result.shoe_type.value} ({result.confidence:.2%})")
        else:
            print(f"✗ Failed to load model: {model_path}")
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  To use this feature, place your trained .tflite model at:")
        print(f"  {os.path.abspath(model_path)}")


def demo_model_switching():
    """Demo switching models at runtime."""
    print("\n" + "="*60)
    print("DEMO 4: Runtime Model Switching")
    print("="*60)
    
    # Start with built-in
    analyzer = ShoeAnalyzer(language="en")
    print("1. Started with built-in CV method")
    
    # Create test image
    test_image = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 150), (250, 350), (100, 120, 140), -1)
    
    # First analysis
    result1 = analyzer.analyze(test_image)
    print(f"   Result: {result1.shoe_type.value} ({result1.confidence:.2%})")
    
    # Switch to custom model
    def better_predictor(image):
        # Simulate a more accurate model
        return ShoeType.LOW_SHOES, 0.95
    
    custom_model = CustomModelAdapter(better_predictor)
    analyzer.set_external_model(custom_model)
    print("\n2. Switched to external model")
    
    # Second analysis
    result2 = analyzer.analyze(test_image)
    print(f"   Result: {result2.shoe_type.value} ({result2.confidence:.2%})")
    print(f"   Confidence improved: {result1.confidence:.2%} → {result2.confidence:.2%}")


def demo_fallback_behavior():
    """Demo fallback when external model fails."""
    print("\n" + "="*60)
    print("DEMO 5: Fallback Behavior (Error Handling)")
    print("="*60)
    
    # Create a faulty predictor that sometimes fails
    def faulty_predictor(image):
        # Simulate model failure
        raise Exception("Model crashed!")
    
    faulty_model = CustomModelAdapter(faulty_predictor)
    analyzer = ShoeAnalyzer(language="en", external_model=faulty_model)
    
    print("Using a model that will crash...")
    
    # Analyze - should fall back gracefully
    test_image = np.zeros((400, 300, 3), dtype=np.uint8)
    result = analyzer.analyze(test_image)
    
    print(f"✓ Fallback successful!")
    print(f"  Detected: {result.shoe_type.value}")
    print(f"  Used built-in CV method after external model failed")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "EXTERNAL MODEL INTEGRATION DEMO" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    
    print("\nThis demo shows how to plug external models into the shoe analyzer.")
    print("The analyzer supports multiple integration methods with automatic fallback.")
    
    try:
        # Run all demos
        demo_without_external_model()
        demo_with_custom_function()
        demo_with_tensorflow_model()
        demo_model_switching()
        demo_fallback_behavior()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ All integration methods demonstrated")
        print("✓ External models improve accuracy and confidence")
        print("✓ Automatic fallback ensures reliability")
        print("✓ Easy to switch models at runtime")
        print("\nNext steps:")
        print("1. Train your own model (see EXTERNAL_MODEL_GUIDE.txt)")
        print("2. Place model in assets/models/ directory")
        print("3. Load it using TensorFlowShoeModel")
        print("4. Integrate into main app (see external_model_example.py)")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
