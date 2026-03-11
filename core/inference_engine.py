"""
Inference engine coordinating all ML model execution.
Manages multi-threaded inference for water and obstacle detection.
"""
import threading
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

from utils.logger import get_logger
from utils.config import ModelConfig, ReflectionConfig
from utils.performance_monitor import get_monitor
from models.water_detector import WaterDetector, WaterDetectionResult
from models.obstacle_detector import ObstacleDetector, ObstacleDetectionResult, BoundingBox
from models.reflection_analyzer import ReflectionAnalyzer, ReflectionResult
from processing.fusion_processor import FusionProcessor, FusedDetectionResult
from models.detection_result import DetectionResult

class InferenceEngine:
    """
    Coordinates ML model inference for detection pipeline.
    Runs water detection and obstacle detection in parallel.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        reflection_config: ReflectionConfig
    ):
        """
        Initialize inference engine.
        
        Args:
            model_config: ML model configuration
            reflection_config: Traditional CV reflection config
        """
        self.logger = get_logger("inference")
        self.monitor = get_monitor()
        self.model_config = model_config
        
        # Initialize detectors
        self.logger.info("Initializing detectors...")
        
        self.water_detector = WaterDetector(model_config, reflection_config)
        self.obstacle_detector = ObstacleDetector(model_config)
        self.reflection_analyzer = ReflectionAnalyzer(reflection_config)
        self.fusion_processor = FusionProcessor()
        
        # Thread pool for parallel inference
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Inference")
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        self.logger.info("InferenceEngine initialized")
    
    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Process a camera frame through all detection models.
        
        Args:
            frame: BGR image from camera (HWC format)
            
        Returns:
            DetectionResult with combined detections, or None on error
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Run water detection and obstacle detection in parallel
            water_future = self._executor.submit(
                self._run_water_detection, frame
            )
            obstacle_future = self._executor.submit(
                self._run_obstacle_detection, frame
            )
            
            # Also run traditional CV reflection analysis
            reflection_future = self._executor.submit(
                self._run_reflection_analysis, frame
            )
            
            # Wait for results
            water_result = water_future.result(timeout=3.0)
            obstacle_result = obstacle_future.result(timeout=3.0)
            reflection_result = reflection_future.result(timeout=3.0)
            
            # Fuse results
            fused = self._fuse_results(
                water_result,
                obstacle_result,
                reflection_result,
                frame.shape
            )
            
            # Convert to DetectionResult
            return DetectionResult(
                water_mask=fused.water_mask,
                water_confidence=fused.water_confidence,
                water_coverage=fused.water_coverage,
                obstacles=fused.obstacles,
                frame_width=frame_w,
                frame_height=frame_h,
                hazard_grid=fused.hazard_grid
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None
    
    def _run_water_detection(
        self,
        frame: np.ndarray
    ) -> Optional[WaterDetectionResult]:
        """Run water detection model."""
        self.monitor.operation_start("water_detection")
        try:
            result = self.water_detector.detect(frame)
            return result
        finally:
            self.monitor.operation_end("water_detection")
    
    def _run_obstacle_detection(
        self,
        frame: np.ndarray
    ) -> Optional[ObstacleDetectionResult]:
        """Run obstacle detection model."""
        self.monitor.operation_start("obstacle_detection")
        try:
            result = self.obstacle_detector.detect(frame)
            return result
        finally:
            self.monitor.operation_end("obstacle_detection")
    
    def _run_reflection_analysis(
        self,
        frame: np.ndarray
    ) -> Optional[ReflectionResult]:
        """Run traditional CV reflection analysis."""
        try:
            return self.reflection_analyzer.analyze(frame)
        except Exception as e:
            self.logger.debug(f"Reflection analysis error: {e}")
            return None
    
    def _fuse_results(
        self,
        water_result: Optional[WaterDetectionResult],
        obstacle_result: Optional[ObstacleDetectionResult],
        reflection_result: Optional[ReflectionResult],
        frame_shape: Tuple[int, ...]
    ) -> FusedDetectionResult:
        """Fuse all detection results."""
        from processing.segmentation_processor import SegmentationResult, PuddleRegion
        from processing.detection_processor import ProcessedDetection
        
        # Convert water result to SegmentationResult format
        seg_result = None
        if water_result is not None:
            seg_result = SegmentationResult(
                binary_mask=water_result.mask,
                confidence_map=water_result.confidence,
                regions=[],  # Simplified
                coverage=water_result.coverage_ratio
            )
        
        # Convert obstacle result to ProcessedDetection format
        det_result = None
        if obstacle_result is not None:
            frame_w = obstacle_result.frame_width
            
            # Count obstacles by position
            left = center = right = 0
            for obs in obstacle_result.obstacles:
                cx = obs.center[0]
                if cx < frame_w / 3:
                    left += 1
                elif cx > 2 * frame_w / 3:
                    right += 1
                else:
                    center += 1
            
            det_result = ProcessedDetection(
                obstacles=obstacle_result.obstacles,
                frame_width=obstacle_result.frame_width,
                frame_height=obstacle_result.frame_height,
                obstacles_left=left,
                obstacles_center=center,
                obstacles_right=right,
                closest_obstacle=None,
                closest_distance=1.0
            )
        
        # Get reflection mask
        reflection_mask = None
        if reflection_result is not None:
            reflection_mask = reflection_result.mask
        
        # Fuse
        return self.fusion_processor.fuse(
            water_result=seg_result,
            obstacle_result=det_result,
            reflection_mask=reflection_mask,
            frame_shape=frame_shape
        )
    
    def process_frame_sync(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Process frame synchronously (single-threaded).
        Use for debugging or when thread pool is unavailable.
        
        Args:
            frame: BGR image
            
        Returns:
            DetectionResult or None
        """
        if frame is None:
            return None
        
        frame_h, frame_w = frame.shape[:2]
        
        # Run detections sequentially
        water_result = self.water_detector.detect(frame)
        obstacle_result = self.obstacle_detector.detect(frame)
        reflection_result = self.reflection_analyzer.analyze(frame)
        
        # Fuse
        fused = self._fuse_results(
            water_result,
            obstacle_result,
            reflection_result,
            frame.shape
        )
        
        return DetectionResult(
            water_mask=fused.water_mask,
            water_confidence=fused.water_confidence,
            water_coverage=fused.water_coverage,
            obstacles=fused.obstacles,
            frame_width=frame_w,
            frame_height=frame_h,
            hazard_grid=fused.hazard_grid
        )
    
    def warm_up(self, frame_shape: Tuple[int, int, int] = (480, 640, 3)):
        """
        Warm up models with dummy inference.
        
        Args:
            frame_shape: Shape of expected frames (H, W, C)
        """
        self.logger.info("Warming up models...")
        
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        
        try:
            self.process_frame_sync(dummy_frame)
            self.logger.info("Model warm-up complete")
        except Exception as e:
            self.logger.warning(f"Warm-up error: {e}")
    
    def release(self):
        """Release all resources."""
        self.logger.info("Releasing inference engine resources")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=False)
        
        # Release detectors
        if self.water_detector:
            self.water_detector.release()
        
        if self.obstacle_detector:
            self.obstacle_detector.release()
        
        self.logger.info("Inference engine released")
