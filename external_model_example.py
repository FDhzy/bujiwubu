"""
Example: How to integrate external shoe recognition models

This file demonstrates three ways to plug in your own shoe recognition model:
1. Using a TensorFlow/TFLite model
2. Using a custom prediction function
3. Creating your own model class

"""
import cv2
import numpy as np
from models.shoe_analyzer import (
    ShoeAnalyzer, 
    ShoeType,
    TensorFlowShoeModel,
    CustomModelAdapter,
    ExternalShoeModel
)


# =============================================================================
# METHOD 1: Using TensorFlow/TFLite Model
# =============================================================================

def example_tensorflow_model():
    """
    Example: Load and use a TensorFlow Lite model for shoe recognition.
    
    Steps to prepare your model:
    1. Train your model to classify 5 shoe types (sandals, low_shoes, ankle_shoes, boots, high_boots)
    2. Convert to TFLite format
    3. Save as .tflite file
    4. Load it using TensorFlowShoeModel
    """
    
    # Create TF model wrapper
    tf_model = TensorFlowShoeModel()
    
    # Load your trained model (replace with your model path)
    model_path = "path/to/your_shoe_model.tflite"
    
    if tf_model.load_model(model_path):
        # Create analyzer with external model
        analyzer = ShoeAnalyzer(language="en", external_model=tf_model)
        
        # Use it to analyze shoes
        image = cv2.imread("test_shoe.jpg")
        result = analyzer.analyze(image)
        
        print(f"Detected: {result.shoe_type.value}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Safe puddle depth: {result.safe_puddle_depth_cm:.1f} cm")
    else:
        print("Failed to load model")


# =============================================================================
# METHOD 2: Using a Custom Prediction Function
# =============================================================================

def my_custom_shoe_predictor(image: np.ndarray):
    """
    Your custom prediction logic.
    This could use any ML framework: PyTorch, ONNX, scikit-learn, etc.
    
    Args:
        image: BGR image from OpenCV
        
    Returns:
        tuple: (ShoeType, confidence_score)
    """
    # Example: Simple rule-based predictor
    # Replace this with your actual ML model inference
    
    height, width = image.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = height / width
    
    # Simple classification based on aspect ratio
    if aspect_ratio < 0.5:
        return ShoeType.SANDALS, 0.75
    elif aspect_ratio < 0.9:
        return ShoeType.LOW_SHOES, 0.80
    elif aspect_ratio < 1.3:
        return ShoeType.ANKLE_SHOES, 0.85
    elif aspect_ratio < 1.8:
        return ShoeType.BOOTS, 0.82
    else:
        return ShoeType.HIGH_BOOTS, 0.78


def example_custom_function():
    """
    Example: Use a custom prediction function.
    """
    # Wrap your custom function
    custom_model = CustomModelAdapter(my_custom_shoe_predictor)
    
    # Create analyzer with custom model
    analyzer = ShoeAnalyzer(language="en", external_model=custom_model)
    
    # Use it
    image = cv2.imread("test_shoe.jpg")
    result = analyzer.analyze(image)
    
    print(f"Detected: {result.shoe_type.value}")
    print(f"Confidence: {result.confidence:.2%}")


# =============================================================================
# METHOD 3: Creating Your Own Model Class
# =============================================================================

class MyONNXShoeModel(ExternalShoeModel):
    """
    Example: Custom model class for ONNX models.
    You can create similar classes for PyTorch, other frameworks.
    """
    
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        
    def load_model(self, model_path: str) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"Loaded ONNX model: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return False
    
    def predict(self, image: np.ndarray):
        """Run ONNX inference."""
        if self.session is None:
            return ShoeType.UNKNOWN, 0.0
        
        try:
            # Preprocess image
            img_resized = cv2.resize(image, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: img_input})
            predictions = outputs[0][0]
            
            # Map to shoe type
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            shoe_types = [
                ShoeType.SANDALS,
                ShoeType.LOW_SHOES,
                ShoeType.ANKLE_SHOES,
                ShoeType.BOOTS,
                ShoeType.HIGH_BOOTS
            ]
            
            return shoe_types[class_idx], confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return ShoeType.UNKNOWN, 0.0


def example_custom_class():
    """
    Example: Use a custom model class (ONNX).
    """
    # Create your custom model
    onnx_model = MyONNXShoeModel()
    
    # Load model
    if onnx_model.load_model("path/to/your_model.onnx"):
        # Create analyzer
        analyzer = ShoeAnalyzer(language="en", external_model=onnx_model)
        
        # Use it
        image = cv2.imread("test_shoe.jpg")
        result = analyzer.analyze(image)
        
        print(f"Detected: {result.shoe_type.value}")


# =============================================================================
# METHOD 4: Runtime Model Switching
# =============================================================================

def example_switch_models():
    """
    Example: Switch between different models at runtime.
    """
    # Start with built-in CV method
    analyzer = ShoeAnalyzer(language="en")
    
    # Later, plug in an external model
    tf_model = TensorFlowShoeModel()
    if tf_model.load_model("path/to/model.tflite"):
        analyzer.set_external_model(tf_model)
        print("Switched to TensorFlow model")
    
    # Or switch to another model
    custom_model = CustomModelAdapter(my_custom_shoe_predictor)
    analyzer.set_external_model(custom_model)
    print("Switched to custom model")


# =============================================================================
# INTEGRATION WITH MAIN APP
# =============================================================================

def integrate_with_app():
    """
    How to integrate your external model with the main application.
    """
    
    # Modify the app initialization in main.py or ui/shoe_setup_screen.py
    
    # Option 1: Initialize with external model from the start
    """
    # In your UI code:
    external_model = TensorFlowShoeModel()
    external_model.load_model("assets/models/my_shoe_classifier.tflite")
    
    self.shoe_analyzer = ShoeAnalyzer(
        language=app_config.language,
        external_model=external_model
    )
    """
    
    # Option 2: Add it later
    """
    # Create analyzer normally
    self.shoe_analyzer = ShoeAnalyzer(language=app_config.language)
    
    # Load external model if available
    try:
        external_model = TensorFlowShoeModel()
        if external_model.load_model("assets/models/my_shoe_classifier.tflite"):
            self.shoe_analyzer.set_external_model(external_model)
    except Exception as e:
        print(f"External model not available: {e}")
    """
    
    pass


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

def prepare_training_data():
    """
    Guidelines for preparing your training data.
    
    You'll need images of shoes in 5 categories:
    
    1. SANDALS (凉鞋)
       - Open shoes, flip-flops, slides
       - ~100-500 images
    
    2. LOW_SHOES (低帮鞋)
       - Sneakers, casual shoes, loafers
       - ~100-500 images
    
    3. ANKLE_SHOES (踝靴)
       - Ankle boots, high-top sneakers
       - ~100-500 images
    
    4. BOOTS (靴子)
       - Mid-calf boots, hiking boots
       - ~100-500 images
    
    5. HIGH_BOOTS (高筒靴/雨靴)
       - Rain boots, knee-high boots
       - ~100-500 images
    
    Image requirements:
    - Various angles (side view most important)
    - Different lighting conditions
    - Multiple colors and styles per category
    - Clear shoe visibility
    - Resolution: 224x224 or higher
    
    Recommended frameworks for training:
    - TensorFlow/Keras: MobileNetV2, EfficientNet
    - PyTorch: ResNet, MobileNet
    - Export to TFLite for mobile deployment
    """
    pass


if __name__ == '__main__':
    print("External Model Integration Examples")
    print("=" * 60)
    print("\nThis file shows how to integrate your own ML model.")
    print("Choose one of the methods above and implement it.")
    print("\nKey steps:")
    print("1. Train your model on shoe images (5 classes)")
    print("2. Export to TFLite, ONNX, or keep in original format")
    print("3. Create a model wrapper (use examples above)")
    print("4. Pass it to ShoeAnalyzer on initialization")
    print("\nThe analyzer will automatically use your model")
    print("and fall back to CV method if it fails.")
    print("=" * 60)
