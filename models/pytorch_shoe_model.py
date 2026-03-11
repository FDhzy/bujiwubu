"""
TFLite-based Shoe Recognition Model
Uses TFLite converted shoe classification model
"""
import os
import hashlib
import json
import numpy as np
import cv2
from typing import Tuple, Optional

from models.shoe_analyzer import ShoeType, ExternalShoeModel
from models.shoe_database import ShoeDatabase
from utils.logger import get_logger

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TFLite not available. Shoe recognition disabled.")


class PyTorchShoeModel(ExternalShoeModel):
    """
    TFLite-based shoe recognition using converted ResNet50 model.
    Integrates with shoe recognition database for caching and storage.
    """

    INPUT_SIZE = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, use_database: bool = True):
        """
        Initialize TFLite shoe model.

        Args:
            use_database: Whether to use database for caching results
        """
        self.logger = get_logger("pytorch_shoe_model")
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.use_database = use_database
        self.database = None

        if use_database:
            try:
                self.database = ShoeDatabase()
                self.logger.info("Database connection established")
            except Exception as e:
                self.logger.warning(f"Failed to connect to database: {e}")
                self.use_database = False

    def load_model(self, model_path: str) -> bool:
        """
        Load TFLite model from file.

        Args:
            model_path: Path to TFLite model file (.tflite)

        Returns:
            True if loaded successfully
        """
        if not TFLITE_AVAILABLE:
            self.logger.error("TFLite not available")
            return False

        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False

            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info(f"Loaded TFLite shoe model from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        Predict shoe type from image.

        Args:
            image: BGR image from OpenCV

        Returns:
            Tuple of (ShoeType, confidence)
        """
        if self.interpreter is None:
            self.logger.warning("Model not loaded, using fallback")
            return ShoeType.UNKNOWN, 0.0

        try:
            # Calculate image hash for caching
            image_hash = None
            if self.use_database and self.database:
                image_hash = self._calculate_image_hash(image)

                # Check cache first
                cached_result = self.database.get_from_cache(image_hash)
                if cached_result:
                    shoe_type_str, confidence = cached_result
                    shoe_type = ShoeType(shoe_type_str)
                    self.logger.debug(f"Using cached result: {shoe_type.value}")
                    return shoe_type, confidence

            # 预处理图像: BGR→RGB, resize(256)后 center_crop(224), 归一化
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # resize to 256x256, then center crop to 224x224
            img_resized = cv2.resize(img_rgb, (256, 256))
            offset = (256 - self.INPUT_SIZE) // 2
            img_crop = img_resized[offset:offset+self.INPUT_SIZE, offset:offset+self.INPUT_SIZE]
            img_float = img_crop.astype(np.float32) / 255.0
            img_norm = (img_float - self.MEAN) / self.STD
            # HWC → CHW → batch: [1, 3, 224, 224]
            input_data = np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0)

            # TFLite 推理
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])  # [1, 5]

            # softmax + argmax
            logits = output[0]  # [5]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])

            # Map prediction to shoe type
            shoe_type = self._map_prediction_to_shoe_type(predicted_idx, confidence)

            self.logger.info(f"Predicted: {shoe_type.value} (confidence={confidence:.2%})")

            # Save to database
            if self.use_database and self.database:
                self.database.save_recognition(
                    shoe_type.value,
                    confidence,
                    image_hash=image_hash
                )

            return shoe_type, confidence

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return ShoeType.UNKNOWN, 0.0

    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """Calculate hash of image for caching."""
        small = cv2.resize(image, (64, 64))
        image_bytes = small.tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def _map_prediction_to_shoe_type(self, predicted_idx: int, confidence: float) -> ShoeType:
        """
        Map model prediction index to ShoeType.

        Args:
            predicted_idx: Index predicted by model (0-4)
            confidence: Confidence score

        Returns:
            ShoeType
        """
        if confidence < 0.3:
            return ShoeType.UNKNOWN

        # Direct mapping from model output to ShoeType
        # 0: sandals, 1: low_shoes, 2: ankle_shoes, 3: boots, 4: high_boots
        mapping = {
            0: ShoeType.SANDALS,
            1: ShoeType.LOW_SHOES,
            2: ShoeType.ANKLE_SHOES,
            3: ShoeType.BOOTS,
            4: ShoeType.HIGH_BOOTS,
        }

        return mapping.get(predicted_idx, ShoeType.UNKNOWN)

    def get_statistics(self):
        """Get recognition statistics from database."""
        if self.database:
            return self.database.get_statistics()
        return {'total': 0}

    def close(self):
        """Close database connection."""
        if self.database:
            self.database.close()

    def __del__(self):
        """Cleanup."""
        self.close()

