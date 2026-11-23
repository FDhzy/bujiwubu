"""
Detection output processor for obstacle detection.
Post-processes YOLO outputs to get clean bounding boxes.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from models.detection_result import DetectionResult
from utils.logger import get_logger
from models.obstacle_detector import BoundingBox, COCO_CLASSES


@dataclass
class ProcessedDetection:
    """Container for processed detection results."""
    # List of obstacle bounding boxes
    obstacles: List[BoundingBox]
    # Frame dimensions
    frame_width: int
    frame_height: int
    # Number of obstacles in each zone
    obstacles_left: int
    obstacles_center: int
    obstacles_right: int
    # Closest obstacle info
    closest_obstacle: Optional[BoundingBox]
    closest_distance: float  # Relative to frame height


class DetectionProcessor:
    """
    Processes YOLO object detection output.
    Parses bounding boxes, applies NMS, and categorizes obstacles.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.45,
        obstacle_classes: Optional[List[int]] = None
    ):
        """
        Initialize detection processor.
        
        Args:
            confidence_threshold: Minimum confidence for valid detection
            nms_threshold: IoU threshold for NMS
            obstacle_classes: List of class IDs to consider as obstacles
        """
        self.logger = get_logger("detection_processor")
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Default obstacle classes if not specified
        self.obstacle_classes = set(obstacle_classes or [
            0,   # person
            1,   # bicycle
            2,   # car
            3,   # motorcycle
            5,   # bus
            7,   # truck
            # 9,   # traffic light - disabled (not relevant for pedestrian navigation)
            10,  # fire hydrant
            11,  # stop sign
            13,  # bench
            56,  # chair
            57,  # couch
            60,  # dining table
        ])
    
    def process(
        self,
        output: np.ndarray,
        frame_size: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int]
    ) -> DetectionResult:
        """
        Process YOLO model output.
        
        Args:
            output: Raw model output tensor
            frame_size: (width, height) of original frame
            scale: Scale factor used in preprocessing
            padding: (pad_x, pad_y) used in preprocessing
            
        Returns:
            ProcessedDetection with categorized obstacles
        """
        frame_w, frame_h = frame_size
        
        # Parse raw output to boxes
        boxes = self._parse_output(output, frame_size, scale, padding)
        
        # Apply NMS
        boxes = self._apply_nms(boxes)
        
        # Filter by obstacle classes
        obstacles = [b for b in boxes if b.class_id in self.obstacle_classes]
        
        # Categorize by position
        left_count = 0
        center_count = 0
        right_count = 0
        
        for obs in obstacles:
            center_x = obs.center[0]
            if center_x < frame_w / 3:
                left_count += 1
            elif center_x > 2 * frame_w / 3:
                right_count += 1
            else:
                center_count += 1
        
        # Find closest obstacle (by bottom edge, closer to camera)
        closest = None
        closest_dist = 1.0
        
        for obs in obstacles:
            # Distance is relative to frame (0 = bottom, 1 = top)
            bottom_y = obs.y + obs.height
            distance = 1.0 - (bottom_y / frame_h)
            
            if distance < closest_dist:
                closest_dist = distance
                closest = obs
        
        return DetectionResult(
            obstacles=obstacles,
            frame_width=frame_w,
            frame_height=frame_h,
            obstacles_left=left_count,
            obstacles_center=center_count,
            obstacles_right=right_count,
            closest_obstacle=closest,
            closest_distance=closest_dist
        )
    
    def _parse_output(
        self,
        output: np.ndarray,
        frame_size: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int]
    ) -> List[BoundingBox]:
        """Parse raw YOLO output to bounding boxes."""
        frame_w, frame_h = frame_size
        pad_x, pad_y = padding
        
        # Handle different YOLO output formats
        if len(output.shape) == 3:
            if output.shape[1] == 85 or output.shape[1] == 84:
                # Shape: (1, 85, num_detections) - transpose
                output = np.transpose(output, (0, 2, 1))
            predictions = output[0]
        else:
            predictions = output
        
        boxes = []
        
        for pred in predictions:
            if len(pred) < 5:
                continue
            
            # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
            x_center, y_center, w, h = pred[:4]
            class_scores = pred[4:]
            
            if len(class_scores) == 0:
                continue
            
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            if confidence < self.conf_threshold:
                continue
            
            # Convert from model coordinates to original frame coordinates
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            w = w / scale
            h = h / scale
            
            # Convert to corner format
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            w = int(w)
            h = int(h)
            
            # Clip to frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            
            if w <= 0 or h <= 0:
                continue
            
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            boxes.append(BoundingBox(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            ))
        
        return boxes
    
    def _apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression."""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < self.nms_threshold]
        
        return keep
    
    def _iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
