"""
Shoe analyzer for determining puddle warning thresholds.
Analyzes uploaded shoe photos to estimate safe water depth.
Supports external model integration for improved accuracy.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from models.shoe_database import ShoeDatabase
from utils.logger import get_logger

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import TensorFlow for external model support
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ShoeType(Enum):
    """Types of shoes with different water resistance."""
    UNKNOWN = "unknown"
    SANDALS = "sandals"           # Open shoes, very low tolerance
    LOW_SHOES = "low_shoes"       # Sneakers, casual shoes
    ANKLE_SHOES = "ankle_shoes"   # Ankle boots, high-top sneakers
    BOOTS = "boots"               # Mid-calf boots
    HIGH_BOOTS = "high_boots"     # Knee-high boots, rain boots


@dataclass
class ShoeAnalysisResult:
    """Result from shoe analysis."""
    # Detected shoe type
    shoe_type: ShoeType
    # Estimated shoe height in cm (from ground to top)
    estimated_height_cm: float
    # Safe puddle depth in cm (water that won't get into shoe)
    safe_puddle_depth_cm: float
    # Warning threshold (0-1, fraction of frame height for puddle)
    puddle_warning_threshold: float
    # Critical threshold (must avoid)
    puddle_critical_threshold: float
    # Confidence in detection (0-1)
    confidence: float
    # Description for user
    description: str


# Shoe type configurations
SHOE_CONFIGS = {
    ShoeType.SANDALS: {
        "height_cm": 3.0,
        "safe_depth_cm": 0.5,
        "description_en": "Sandals detected. Will warn for very shallow puddles.",
        "description_zh": "已选择凉鞋，将对极浅水坑发出警告。",
    },
    ShoeType.LOW_SHOES: {
        "height_cm": 8.0,
        "safe_depth_cm": 2.0,
        "description_en": "Low shoes detected. Will warn for shallow to medium puddles.",
        "description_zh": "已选择低帮鞋，将对浅至中等水坑发出警告。",
    },
    ShoeType.ANKLE_SHOES: {
        "height_cm": 12.0,
        "safe_depth_cm": 5.0,
        "description_en": "Ankle shoes detected. Will warn for medium puddles.",
        "description_zh": "已选择踝靴，将对中等深度水坑发出警告。",
    },
    ShoeType.BOOTS: {
        "height_cm": 25.0,
        "safe_depth_cm": 15.0,
        "description_en": "Boots detected. Will warn for deep puddles only.",
        "description_zh": "已选择靴子，仅对较深水坑发出警告。",
    },
    ShoeType.HIGH_BOOTS: {
        "height_cm": 40.0,
        "safe_depth_cm": 25.0,
        "description_en": "High boots/rain boots detected. High water resistance.",
        "description_zh": "已选择雨靴，防水性能优异。",
    },
    ShoeType.UNKNOWN: {
        "height_cm": 8.0,
        "safe_depth_cm": 2.0,
        "description_en": "Could not identify shoe type. Using default settings.",
        "description_zh": "无法识别鞋型，使用默认设置。",
    },
}


class ExternalShoeModel(ABC):
    """
    Abstract base class for external shoe recognition models.
    Implement this interface to plug in your own ML model.
    """
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        Predict shoe type from image.
        
        Args:
            image: BGR image of the shoe
            
        Returns:
            Tuple of (ShoeType, confidence score 0-1)
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load the external model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        pass


class TensorFlowShoeModel(ExternalShoeModel):
    """
    TensorFlow/TFLite model wrapper for shoe recognition.
    Example implementation for external model integration.
    """
    
    def __init__(self):
        """Initialize TensorFlow model wrapper."""
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger = get_logger("tf_shoe_model")
        
        # Class mapping (customize based on your model)
        self.class_mapping = {
            0: ShoeType.SANDALS,
            1: ShoeType.LOW_SHOES,
            2: ShoeType.ANKLE_SHOES,
            3: ShoeType.BOOTS,
            4: ShoeType.HIGH_BOOTS
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        Load TFLite or SavedModel format.
        
        Args:
            model_path: Path to .tflite or saved_model directory
            
        Returns:
            True if loaded successfully
        """
        try:
            if not TF_AVAILABLE:
                self.logger.error("TensorFlow not available")
                return False
            
            # Check if it's a TFLite model
            if model_path.endswith('.tflite'):
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.logger.info(f"Loaded TFLite model from {model_path}")
            else:
                # Load as SavedModel
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"Loaded Keras model from {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        Run inference on shoe image.
        
        Args:
            image: BGR image
            
        Returns:
            (ShoeType, confidence)
        """
        if self.model is None and self.interpreter is None:
            return ShoeType.UNKNOWN, 0.0
        
        try:
            # Preprocess image
            if self.interpreter:
                # TFLite model
                input_shape = self.input_details[0]['shape']
                input_size = (input_shape[2], input_shape[1])  # (width, height)
            else:
                # Keras model
                input_shape = self.model.input_shape
                input_size = (input_shape[2], input_shape[1])
            
            # Resize and normalize
            img_resized = cv2.resize(image, input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)
            
            # Run inference
            if self.interpreter:
                # TFLite inference
                self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            else:
                # Keras inference
                output = self.model.predict(img_input, verbose=0)[0]
            
            # Get prediction
            class_idx = np.argmax(output)
            confidence = float(output[class_idx])
            
            shoe_type = self.class_mapping.get(class_idx, ShoeType.UNKNOWN)
            
            return shoe_type, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return ShoeType.UNKNOWN, 0.0


class CustomModelAdapter(ExternalShoeModel):
    """
    Adapter for custom prediction functions.
    Use this to wrap any custom prediction logic.
    """
    
    def __init__(self, predict_fn: Callable[[np.ndarray], Tuple[ShoeType, float]]):
        """
        Initialize with a custom prediction function.
        
        Args:
            predict_fn: Function that takes image and returns (ShoeType, confidence)
        """
        self.predict_fn = predict_fn
        self.logger = get_logger("custom_model")
    
    def load_model(self, model_path: str) -> bool:
        """Not needed for custom function adapter."""
        return True
    
    def predict(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        Run custom prediction function.
        
        Args:
            image: BGR image
            
        Returns:
            (ShoeType, confidence)
        """
        try:
            return self.predict_fn(image)
        except Exception as e:
            self.logger.error(f"Custom prediction error: {e}")
            return ShoeType.UNKNOWN, 0.0


class ShoeAnalyzer:
    """
    Analyzes shoe photos to determine puddle warning thresholds.
    Supports both built-in CV methods and external ML models.
    Implements stability measures to ensure consistent recognition.
    """
    
    def __init__(self, language: str = "en", external_model: Optional[ExternalShoeModel] = None):
        """
        Initialize shoe analyzer.
        
        Args:
            language: Language for descriptions ("en" or "zh")
            external_model: Optional external ML model for improved recognition
        """
        self.logger = get_logger("shoe_analyzer")
        self.language = language
        self.external_model = external_model
        
        # Feature extractor for database matching
        self.orb = cv2.ORB_create(nfeatures=500) if CV2_AVAILABLE else None
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if CV2_AVAILABLE else None
        
        # Database connection
        try:
            self.database = ShoeDatabase()
        except Exception:
            self.database = None
            
        # Face detector to prevent misidentifying faces as shoes
        self.face_cascade = None
        if CV2_AVAILABLE:
            try:
                import os
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    self.logger.info("Face detector initialized for shoe recognition safety")
            except Exception as e:
                self.logger.warning(f"Failed to initialize face detector: {e}")

        # Reference for size estimation (average foot length ~25cm)
        self.reference_foot_length_cm = 25.0
        
        # Stability cache for consistent recognition
        self._last_detection_cache = None  # Cache last detection
        self._detection_history = []  # History of recent detections
        self._max_history_size = 5  # Keep last 5 detections for stability
        
        if external_model:
            self.logger.info("ShoeAnalyzer initialized with external model")
        else:
            self.logger.info("ShoeAnalyzer initialized with built-in CV")
    
    def set_external_model(self, model: ExternalShoeModel) -> None:
        """
        Set or update the external recognition model.
        
        Args:
            model: External model instance
        """
        self.external_model = model
        self.logger.info("External model updated")
    
    def reset_detection_cache(self) -> None:
        """
        Reset detection cache and history.
        Call this when starting a new shoe analysis session.
        """
        self._last_detection_cache = None
        self._detection_history = []
        self.logger.debug("Detection cache reset")
    
    def _is_likely_skin(self, image: np.ndarray) -> bool:
        """Detect if the image contains large amounts of skin tone."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Standard skin tone range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(mask > 0) / mask.size
            
            # If more than 40% of the image is skin tone, it's likely not a shoe
            return skin_ratio > 0.4
        except Exception:
            return False

    def _match_with_database(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """Match image features against database templates."""
        try:
            # Extract features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            
            if des is None or len(des) < 10:
                return ShoeType.UNKNOWN, 0.0
            
            templates = self.database.get_all_templates()
            if not templates:
                return ShoeType.UNKNOWN, 0.0
            
            best_type = ShoeType.UNKNOWN
            best_score = 0.0
            
            for shoe_type_str, template_des in templates:
                # Match descriptors
                matches = self.bf_matcher.match(des, template_des)
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Use top 20 matches to calculate score
                good_matches = [m for m in matches[:20] if m.distance < 50]
                score = len(good_matches) / 20.0
                
                if score > best_score:
                    best_score = score
                    try:
                        best_type = ShoeType(shoe_type_str)
                    except ValueError:
                        best_type = ShoeType.UNKNOWN
            
            return best_type, best_score
            
        except Exception as e:
            self.logger.error(f"Database matching error: {e}")
            return ShoeType.UNKNOWN, 0.0

    def analyze(self, image: np.ndarray) -> ShoeAnalysisResult:
        """
        Analyze shoe image to determine type and safe puddle depth.
        Uses external model if available, falls back to CV method.
        
        Args:
            image: BGR image of the shoe
            
        Returns:
            ShoeAnalysisResult with thresholds
        """
        if image is None or image.size == 0:
            return self._create_default_result()
        
        try:
            # Try external model first if available
            if self.external_model:
                try:
                    shoe_type, confidence = self.external_model.predict(image)
                    self.logger.info(f"External model prediction: {shoe_type.value} ({confidence:.2%})")
                    
                    # If confidence is low, fall back to CV method
                    if confidence < 0.15:
                        self.logger.warning("Low confidence, falling back to CV method")
                        shoe_type, confidence = self._detect_shoe_type(image)
                except Exception as e:
                    self.logger.error(f"External model failed: {e}, using CV method")
                    shoe_type, confidence = self._detect_shoe_type(image)
            else:
                # Use built-in CV method
                shoe_type, confidence = self._detect_shoe_type(image)
            
            # Estimate actual height from image
            estimated_height = self._estimate_height(image, shoe_type)
            
            # Calculate thresholds
            config = SHOE_CONFIGS[shoe_type]
            safe_depth = config["safe_depth_cm"]
            
            # Adjust based on estimated height if different from default
            if estimated_height > 0:
                height_ratio = estimated_height / config["height_cm"]
                safe_depth = config["safe_depth_cm"] * min(height_ratio, 1.5)
            
            # Convert to frame-relative thresholds
            # Assume puddle that covers bottom 20% of frame is ~10cm deep
            cm_per_frame_ratio = 50.0  # 50cm per full frame height at ground level
            
            warning_threshold = safe_depth / cm_per_frame_ratio
            critical_threshold = (safe_depth * 0.5) / cm_per_frame_ratio
            
            # Get description in correct language
            desc_key = "description_zh" if self.language == "zh" else "description_en"
            description = config[desc_key]
            
            # Save as template if confidence is high and it's not unknown
            if self.database and self.orb and confidence > 0.7 and shoe_type != ShoeType.UNKNOWN:
                self._save_image_template(image, shoe_type)
            
            result = ShoeAnalysisResult(
                shoe_type=shoe_type,
                estimated_height_cm=estimated_height if estimated_height > 0 else config["height_cm"],
                safe_puddle_depth_cm=safe_depth,
                puddle_warning_threshold=warning_threshold,
                puddle_critical_threshold=critical_threshold,
                confidence=confidence,
                description=description
            )
            
            self.logger.info(
                f"Shoe analysis: {shoe_type.value}, "
                f"safe depth={safe_depth:.1f}cm, "
                f"confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Shoe analysis error: {e}")
            return self._create_default_result()
    
    def _detect_shoe_type(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        Detect shoe type from image using visual features and database templates.
        Includes face rejection safety check.
        """
        if not CV2_AVAILABLE:
            return ShoeType.UNKNOWN, 0.5
        
        h, w = image.shape[:2]

        # 1. Safety Check: Face Detection
        # If a face is clearly visible, it's definitely not a shoe
        if self.face_cascade is not None:
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_full, 1.3, 5)
            if len(faces) > 0:
                self.logger.warning("Face detected in shoe capture frame! Rejecting detection.")
                return ShoeType.UNKNOWN, 0.0

        # 2. Safety Check: Skin Tone detection
        # Check if the image contains large areas of skin tone (likely a face or limb)
        if self._is_likely_skin(image):
            self.logger.warning("Large skin tone area detected! Rejecting detection.")
            return ShoeType.UNKNOWN, 0.0

        # 3. Feature-Based Matching via Database
        if self.orb and self.database:
            db_type, db_conf = self._match_with_database(image)
            if db_conf > 0.6:
                self.logger.info(f"Database match found: {db_type.value} ({db_conf:.2%})")
                return db_type, db_conf

        # 2. Heuristic CV Fallback (improved from before)
        h, w = image.shape[:2]
        # ... rest of existing CV logic
        
        # Preprocessing for stability - normalize brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for consistent brightness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise for more stable edge detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Use more stable edge detection parameters
        edges = cv2.Canny(gray, 40, 120)  # Slightly adjusted thresholds
        
        # Morphological operations to stabilize contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return cached result if available
            if self._last_detection_cache:
                self.logger.debug("No contours found, using cached result")
                return self._last_detection_cache
            return ShoeType.UNKNOWN, 0.3
        
        # Find largest contour (likely the shoe)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        x, y, cw, ch = cv2.boundingRect(largest_contour)

        # 3. Position and Scale Heuristics
        # - Shoes should ideally be in the lower part of the frame during capture
        # - Shoes shouldn't take up the entire frame (usually)
        if y < h * 0.1: # Too close to top edge
            self.logger.debug("Contour too high in frame, likely background or face")
            return ShoeType.UNKNOWN, 0.2

        # Filter out tiny contours (noise)
        if contour_area < (h * w * 0.05):  # Increased from 3% to 5% for better noise rejection
            if self._last_detection_cache:
                self.logger.debug("Contour too small, using cached result")
                return self._last_detection_cache
            return ShoeType.UNKNOWN, 0.3
        
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio with rounding for stability
        aspect_ratio = round(ch / cw, 2) if cw > 0 else 1.0
        
        # Extract shoe region
        shoe_region = gray[max(0, y):min(h, y+ch), max(0, x):min(w, x+cw)]
        
        if shoe_region.size == 0:
            if self._last_detection_cache:
                return self._last_detection_cache
            return ShoeType.UNKNOWN, 0.3
        
        # Calculate features with stability measures
        laplacian_var = cv2.Laplacian(shoe_region, cv2.CV_64F).var()
        color_std = np.std(shoe_region)
        
        # Classify with more robust thresholds and hysteresis
        # If we have a previous detection, use hysteresis to prevent flickering
        if self._last_detection_cache:
            prev_type, prev_conf = self._last_detection_cache
            # Add hysteresis zones around boundaries
            hysteresis = 0.05
            
            # Check for sandals
            if aspect_ratio < 0.45 - hysteresis and laplacian_var > 400:
                shoe_type = ShoeType.SANDALS
                confidence = 0.7
            elif aspect_ratio < 0.45 + hysteresis and prev_type == ShoeType.SANDALS:
                shoe_type = ShoeType.SANDALS
                confidence = 0.65
            
            # Check boundaries with hysteresis
            elif aspect_ratio < 0.55 - hysteresis:
                shoe_type = ShoeType.SANDALS
                confidence = 0.6
            elif aspect_ratio < 0.55 + hysteresis and prev_type == ShoeType.SANDALS:
                shoe_type = ShoeType.SANDALS
                confidence = 0.55
            
            elif aspect_ratio < 0.85 - hysteresis:
                shoe_type = ShoeType.LOW_SHOES
                confidence = 0.7
            elif aspect_ratio < 0.85 + hysteresis and prev_type == ShoeType.LOW_SHOES:
                shoe_type = ShoeType.LOW_SHOES
                confidence = 0.65
            
            elif aspect_ratio < 1.25 - hysteresis:
                shoe_type = ShoeType.ANKLE_SHOES
                confidence = 0.7
            elif aspect_ratio < 1.25 + hysteresis and prev_type == ShoeType.ANKLE_SHOES:
                shoe_type = ShoeType.ANKLE_SHOES
                confidence = 0.65
            
            elif aspect_ratio < 1.85 - hysteresis:
                shoe_type = ShoeType.BOOTS
                confidence = 0.65
            elif aspect_ratio < 1.85 + hysteresis and prev_type == ShoeType.BOOTS:
                shoe_type = ShoeType.BOOTS
                confidence = 0.60
            
            else:
                shoe_type = ShoeType.HIGH_BOOTS
                confidence = 0.6
        else:
            # First detection - use standard thresholds
            if aspect_ratio < 0.4 and laplacian_var > 400:
                shoe_type = ShoeType.SANDALS
                confidence = 0.7
            elif aspect_ratio < 0.55:
                shoe_type = ShoeType.SANDALS
                confidence = 0.6
            elif aspect_ratio < 0.85:
                shoe_type = ShoeType.LOW_SHOES
                confidence = 0.7
            elif aspect_ratio < 1.25:
                shoe_type = ShoeType.ANKLE_SHOES
                confidence = 0.7
            elif aspect_ratio < 1.85:
                shoe_type = ShoeType.BOOTS
                confidence = 0.65
            else:
                shoe_type = ShoeType.HIGH_BOOTS
                confidence = 0.6
        
        # Check for rain boots (uniform color + tall)
        if color_std < 32 and aspect_ratio > 1.1:
            shoe_type = ShoeType.HIGH_BOOTS
            confidence = min(confidence + 0.1, 0.9)
        
        # Add to detection history for temporal smoothing
        self._detection_history.append((shoe_type, confidence, aspect_ratio))
        if len(self._detection_history) > self._max_history_size:
            self._detection_history.pop(0)
        
        # If we have enough history, use majority voting for stability
        if len(self._detection_history) >= 3:
            shoe_type, confidence = self._stabilize_detection(shoe_type, confidence)
        
        # Cache the result
        self._last_detection_cache = (shoe_type, confidence)
        
        self.logger.debug(
            f"Detected: {shoe_type.value}, aspect={aspect_ratio:.2f}, "
            f"conf={confidence:.2f}, color_std={color_std:.1f}"
        )
        
        return shoe_type, confidence
    
    def _stabilize_detection(self, current_type: ShoeType, current_conf: float) -> Tuple[ShoeType, float]:
        """
        Use detection history to stabilize results via majority voting.
        
        Args:
            current_type: Current detection
            current_conf: Current confidence
            
        Returns:
            Stabilized (ShoeType, confidence)
        """
        if len(self._detection_history) < 2:
            return current_type, current_conf
        
        # Count occurrences of each type
        type_counts = {}
        for shoe_type, _, _ in self._detection_history:
            type_counts[shoe_type] = type_counts.get(shoe_type, 0) + 1
        
        # Find most common type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])
        majority_type = most_common_type[0]
        majority_count = most_common_type[1]
        
        # If current type matches majority, keep it
        if current_type == majority_type:
            # Boost confidence slightly for stability
            return current_type, min(current_conf + 0.05, 0.95)
        
        # If majority is strong (>50%), use majority
        if majority_count > len(self._detection_history) / 2:
            avg_conf = sum(c for t, c, _ in self._detection_history if t == majority_type) / majority_count
            self.logger.debug(f"Stabilized to {majority_type.value} (majority vote)")
            return majority_type, avg_conf
        
        # Otherwise stick with current
        return current_type, current_conf
    
    def _estimate_height(self, image: np.ndarray, shoe_type: ShoeType) -> float:
        """
        Estimate actual shoe height from image.
        
        Args:
            image: BGR image
            shoe_type: Detected shoe type
            
        Returns:
            Estimated height in cm, or 0 if unable to estimate
        """
        if not CV2_AVAILABLE:
            return 0.0
        
        h, w = image.shape[:2]
        
        # Try to find shoe contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Assume foot length is visible in image
        # Use reference foot length to estimate scale
        if cw > 0:
            # Estimate pixels per cm based on assumed foot length
            pixels_per_cm = cw / self.reference_foot_length_cm
            
            # Calculate shoe height in cm
            estimated_height = ch / pixels_per_cm
            
            # Sanity check
            if 2.0 < estimated_height < 50.0:
                return estimated_height
        
        return 0.0
    
    def _create_default_result(self) -> ShoeAnalysisResult:
        """Create default result when analysis fails."""
        config = SHOE_CONFIGS[ShoeType.UNKNOWN]
        desc_key = "description_zh" if self.language == "zh" else "description_en"
        
        return ShoeAnalysisResult(
            shoe_type=ShoeType.UNKNOWN,
            estimated_height_cm=config["height_cm"],
            safe_puddle_depth_cm=config["safe_depth_cm"],
            puddle_warning_threshold=0.04,  # Default ~2cm
            puddle_critical_threshold=0.02,
            confidence=0.0,
            description=config[desc_key]
        )
    
    def _save_image_template(self, image: np.ndarray, shoe_type: ShoeType, is_user_verified: bool = False):
        """Save image features as a template in the database."""
        try:
            import hashlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            
            if des is not None and len(des) > 20:
                # Calculate image hash for uniqueness
                small = cv2.resize(image, (64, 64))
                img_hash = hashlib.md5(small.tobytes()).hexdigest()
                
                self.database.save_template(
                    shoe_type.value, 
                    des, 
                    img_hash, 
                    is_user_verified=1 if is_user_verified else 0
                )
        except Exception as e:
            self.logger.debug(f"Failed to save template: {e}")

    def get_shoe_type_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all shoe types for user selection."""
        descriptions = {}
        desc_key = "description_zh" if self.language == "zh" else "description_en"
        
        for shoe_type in ShoeType:
            if shoe_type != ShoeType.UNKNOWN:
                config = SHOE_CONFIGS[shoe_type]
                descriptions[shoe_type.value] = config[desc_key]
        
        return descriptions
    
    def get_result_for_type(self, shoe_type: ShoeType) -> ShoeAnalysisResult:
        """
        Get analysis result for a manually selected shoe type.
        
        Args:
            shoe_type: User-selected shoe type
            
        Returns:
            ShoeAnalysisResult for that type
        """
        config = SHOE_CONFIGS[shoe_type]
        desc_key = "description_zh" if self.language == "zh" else "description_en"
        
        safe_depth = config["safe_depth_cm"]
        cm_per_frame_ratio = 50.0
        
        return ShoeAnalysisResult(
            shoe_type=shoe_type,
            estimated_height_cm=config["height_cm"],
            safe_puddle_depth_cm=safe_depth,
            puddle_warning_threshold=safe_depth / cm_per_frame_ratio,
            puddle_critical_threshold=(safe_depth * 0.5) / cm_per_frame_ratio,
            confidence=1.0,  # Manual selection = full confidence
            description=config[desc_key]
        )
