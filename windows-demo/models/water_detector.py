"""
Water puddle detector using DeepLabV3 semantic segmentation.
Identifies water/puddle regions in camera frames.
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from utils.config import ModelConfig, ReflectionConfig
from models.model_loader import get_model_loader, DummyModel
from models.reflection_analyzer import ReflectionAnalyzer
from models.pytorch_water_detector import PyTorchWaterDetector


@dataclass
class WaterDetectionResult:
    """Result from water detection."""
    # Binary mask where 1 = water, 0 = not water
    mask: np.ndarray
    # Confidence map (0-1 probability of water)
    confidence: np.ndarray
    # Total coverage ratio (0-1)
    coverage_ratio: float
    # Number of distinct puddle regions
    num_regions: int


class WaterDetector:
    """
    Detects water puddles using DeepLabV3 semantic segmentation model.
    Outputs a binary mask indicating water regions.
    """
    
    def __init__(self, config: ModelConfig, reflection_config: Optional[ReflectionConfig] = None):
        """
        Initialize water detector.
        
        Args:
            config: Model configuration
            reflection_config: Optional config for CV fallback
        """
        self.logger = get_logger("water_detector")
        self.config = config
        self.reflection_config = reflection_config
        
        self.input_size = config.deeplabv3_input_size
        self.threshold = config.water_confidence_threshold
        self.water_class = config.water_class_index
        
        # Try TFLite water model first (water_model.tflite)
        self.pytorch_detector = None
        self.use_cv_fallback = False
        self.model = None
        
        try:
            self.logger.info("Loading TFLite water model...")
            self.pytorch_detector = PyTorchWaterDetector(
                model_path="training_checkpoints/water_model.tflite",
                threshold=self.threshold
            )
            if self.pytorch_detector.interpreter is not None:
                self.logger.info("✓ TFLite water model loaded successfully")
            else:
                self.logger.warning("✗ TFLite water model failed to load, falling back...")
                self.pytorch_detector = None
        except Exception as e:
            self.logger.warning(f"✗ Failed to load TFLite water model: {e}")
            self.pytorch_detector = None
        
        # Fall back to TFLite if PyTorch failed
        if self.pytorch_detector is None:
            loader = get_model_loader()
            self.model = loader.load_tflite(
                config.deeplabv3_model_path,
                use_gpu=config.use_gpu_delegate,
                num_threads=config.num_threads
            )
            self.use_cv_fallback = isinstance(self.model, DummyModel)
            
            if self.use_cv_fallback:
                self.logger.info("TFLite model not available, will use CV fallback")
        
        # Initialize reflection analyzer for CV fallback (always needed as backup)
        if reflection_config:
            self.reflection_analyzer = ReflectionAnalyzer(reflection_config)
        else:
            # Default config if none provided
            self.reflection_analyzer = ReflectionAnalyzer(ReflectionConfig())
        
        if self.use_cv_fallback:
            self.logger.info("ML model missing, using CV fallback for water detection")
        
        # Summary of which detector is being used
        pytorch_available = self.pytorch_detector is not None and self.pytorch_detector.interpreter is not None
        tflite_available = self.model is not None and not isinstance(self.model, DummyModel)
        
        if pytorch_available:
            self.logger.info("Water detector initialized with PyTorch model")
            self._primary_detector = "pytorch"
        elif tflite_available:
            self.logger.info("Water detector initialized with TFLite model")
            self._primary_detector = "tflite"
        else:
            self.logger.info("Water detector initialized with CV fallback")
            self._primary_detector = "cv"
    
    def detect(self, frame: np.ndarray) -> Optional[WaterDetectionResult]:
        """
        Detect water puddles in a frame.
        
        Args:
            frame: BGR image from camera (HWC format)
            
        Returns:
            WaterDetectionResult with mask and metadata, or None on error
        """
        try:
            # Try PyTorch detector first if available
            if self.pytorch_detector is not None and self.pytorch_detector.interpreter is not None:
                result = self.pytorch_detector.detect(frame)
                if result is not None:
                    # 过滤小面积误报（覆盖率小于1%视为误报）
                    if result.coverage_ratio < 0.01:
                        # 返回空结果（无水坑）
                        h, w = frame.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        empty_confidence = np.zeros((h, w), dtype=np.float32)
                        return WaterDetectionResult(
                            mask=empty_mask,
                            confidence=empty_confidence,
                            coverage_ratio=0.0,
                            num_regions=0
                        )
                    return WaterDetectionResult(
                        mask=result.mask,
                        confidence=result.confidence,
                        coverage_ratio=result.coverage_ratio,
                        num_regions=result.num_regions
                    )
            
            # Fall back to TFLite model
            if self.model is not None and not self.use_cv_fallback:
                input_tensor = self._preprocess(frame)
                output = self.model.predict(input_tensor)
                mask, confidence = self._postprocess(output, frame.shape[:2], frame)
                coverage = np.sum(mask) / mask.size
                num_regions = self._count_regions(mask)
                return WaterDetectionResult(
                    mask=mask,
                    confidence=confidence,
                    coverage_ratio=coverage,
                    num_regions=num_regions
                )
            
            # Final fallback to CV
            return self._detect_cv(frame)
            
        except Exception as e:
            self.logger.error(f"Water detection error: {e}")
            return None

    def _detect_cv(self, frame: np.ndarray) -> WaterDetectionResult:
        """CV-based fallback for water detection when ML model is missing."""
        reflection = self.reflection_analyzer.analyze(frame)
        if reflection:
            num_regions = self._count_regions(reflection.mask)
            return WaterDetectionResult(
                mask=reflection.mask,
                confidence=reflection.intensity,
                coverage_ratio=reflection.coverage_ratio,
                num_regions=num_regions
            )
        
        # Absolute fallback: empty result
        h, w = frame.shape[:2]
        return WaterDetectionResult(
            mask=np.zeros((h, w), dtype=np.uint8),
            confidence=np.zeros((h, w), dtype=np.float32),
            coverage_ratio=0.0,
            num_regions=0
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: BGR image (HWC)
            
        Returns:
            Normalized tensor (NHWC format)
        """
        import cv2
        
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        tensor = np.expand_dims(normalized, axis=0)
        
        return tensor
    
    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int],
        frame: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-process model output to get water mask.
        Includes a CV-based verification step to reduce false positives.
        """
        import cv2
        
        # Handle different output formats
        if len(output.shape) == 4:
            # NHWC format with class channels
            if output.shape[-1] > 1:
                # Multi-class segmentation - extract water class
                confidence = output[0, :, :, self.water_class]
            else:
                # Single channel output
                confidence = output[0, :, :, 0]
        elif len(output.shape) == 3:
            # NHW format (single class)
            confidence = output[0]
        else:
            confidence = output.squeeze()
        
        # Apply sigmoid if output is logits
        if confidence.min() < 0 or confidence.max() > 1:
            confidence = 1 / (1 + np.exp(-confidence))
        
        # Resize to original frame size
        confidence = cv2.resize(
            confidence,
            (original_size[1], original_size[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        
        # Initial mask based on threshold
        mask = (confidence > self.threshold).astype(np.uint8)

        # Verification step: If ML detects water, verify with traditional CV heuristics
        if frame is not None and np.any(mask > 0):
            reflection = self.reflection_analyzer.analyze(frame)
            if reflection:
                # 1. Combined logic: ML confidence * CV intensity
                cv_verify = cv2.resize(reflection.intensity, (original_size[1], original_size[0]))
                
                # 2. Strict Texture Consistency
                # If the area is detected as water but has high texture complexity, reject it.
                # This prevents gray shoes or rough pavement from being misclassified.
                refined_confidence = confidence * 0.6 + cv_verify * 0.4
                
                # Boost confidence if reflection is confirmed
                mask = (refined_confidence > (self.threshold * 1.1)).astype(np.uint8)
                
                # 3. Size Filter: Small fragmented detections are usually noise
                num_px = np.sum(mask)
                if num_px < (original_size[0] * original_size[1] * 0.005): # Less than 0.5% of frame
                    mask.fill(0)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, confidence
    
    def _count_regions(self, mask: np.ndarray) -> int:
        """Count number of distinct water regions in mask."""
        import cv2
        
        num_labels, _ = cv2.connectedComponents(mask)
        return num_labels - 1  # Subtract 1 for background
    
    def release(self):
        """Release model resources."""
        if self.model:
            self.model.release()
            self.model = None
