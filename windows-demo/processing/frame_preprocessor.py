"""
Frame preprocessing utilities for ML model input.
Handles resizing, normalization, and format conversions.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from utils.logger import get_logger


@dataclass
class PreprocessedFrame:
    """Container for preprocessed frame data."""
    # Preprocessed tensor ready for model input
    tensor: np.ndarray
    # Original frame for reference
    original: np.ndarray
    # Scale factor applied
    scale: float
    # Padding applied (pad_x, pad_y)
    padding: Tuple[int, int]
    # Original dimensions (height, width)
    original_size: Tuple[int, int]


class FramePreprocessor:
    """
    Preprocesses camera frames for ML model inference.
    Handles resizing, color conversion, and normalization.
    """
    
    def __init__(self):
        self.logger = get_logger("preprocessor")
    
    def preprocess_for_segmentation(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ) -> PreprocessedFrame:
        """
        Preprocess frame for segmentation model (DeepLabV3).
        
        Args:
            frame: BGR image (HWC format)
            target_size: (width, height) for model input
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            PreprocessedFrame with tensor and metadata
        """
        import cv2
        
        original_h, original_w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Simple resize (no letterboxing for segmentation)
        resized = cv2.resize(frame, target_size)
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        if normalize:
            tensor = rgb.astype(np.float32) / 255.0
        else:
            tensor = rgb.astype(np.float32)
        
        # Add batch dimension (NHWC)
        tensor = np.expand_dims(tensor, axis=0)
        
        scale = min(target_w / original_w, target_h / original_h)
        
        return PreprocessedFrame(
            tensor=tensor,
            original=frame,
            scale=scale,
            padding=(0, 0),
            original_size=(original_h, original_w)
        )
    
    def preprocess_for_detection(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (320, 320),
        normalize: bool = True
    ) -> PreprocessedFrame:
        """
        Preprocess frame for object detection model (YOLO).
        Uses letterboxing to maintain aspect ratio.
        
        Args:
            frame: BGR image (HWC format)
            target_size: (width, height) for model input
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            PreprocessedFrame with tensor and metadata
        """
        import cv2
        
        original_h, original_w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit while maintaining aspect ratio
        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create letterboxed image (pad with gray)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize
        if normalize:
            tensor = rgb.astype(np.float32) / 255.0
        else:
            tensor = rgb.astype(np.float32)
        
        # Add batch dimension (NHWC)
        tensor = np.expand_dims(tensor, axis=0)
        
        return PreprocessedFrame(
            tensor=tensor,
            original=frame,
            scale=scale,
            padding=(pad_x, pad_y),
            original_size=(original_h, original_w)
        )
    
    def resize_mask_to_original(
        self,
        mask: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize a segmentation mask back to original frame size.
        
        Args:
            mask: Binary or probability mask
            original_size: (height, width) of original frame
            
        Returns:
            Resized mask
        """
        import cv2
        
        return cv2.resize(
            mask.astype(np.float32),
            (original_size[1], original_size[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )
    
    def convert_detection_coords(
        self,
        boxes: np.ndarray,
        preprocessed: PreprocessedFrame
    ) -> np.ndarray:
        """
        Convert detection coordinates from model space to original frame space.
        
        Args:
            boxes: Array of [x, y, w, h] in model coordinates
            preprocessed: PreprocessedFrame with scale and padding info
            
        Returns:
            Array of boxes in original frame coordinates
        """
        if len(boxes) == 0:
            return boxes
        
        converted = boxes.copy().astype(np.float32)
        
        # Remove padding offset
        converted[:, 0] -= preprocessed.padding[0]  # x
        converted[:, 1] -= preprocessed.padding[1]  # y
        
        # Scale back to original size
        converted /= preprocessed.scale
        
        # Clip to frame bounds
        original_h, original_w = preprocessed.original_size
        converted[:, 0] = np.clip(converted[:, 0], 0, original_w)
        converted[:, 1] = np.clip(converted[:, 1], 0, original_h)
        converted[:, 2] = np.clip(converted[:, 2], 0, original_w - converted[:, 0])
        converted[:, 3] = np.clip(converted[:, 3], 0, original_h - converted[:, 1])
        
        return converted.astype(np.int32)
