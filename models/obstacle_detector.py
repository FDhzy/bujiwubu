"""
Obstacle detector using YOLOv8-Nano object detection.
Identifies obstacles like people, vehicles, poles, furniture, etc.
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from utils.config import ModelConfig
from models.model_loader import get_model_loader, DummyModel
from models.pytorch_obstacle_detector import PyTorchObstacleDetector


@dataclass
class BoundingBox:
    """Bounding box for detected obstacle."""
    # Box coordinates (x, y, width, height) in pixels
    x: int
    y: int
    width: int
    height: int
    # Detection confidence (0-1)
    confidence: float
    # Class ID
    class_id: int
    # Class name (optional)
    class_name: str = ""
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bbox as tuple (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bbox."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get area of bbox in pixels."""
        return self.width * self.height


@dataclass
class ObstacleDetectionResult:
    """Result from obstacle detection."""
    # List of detected obstacles
    obstacles: List[BoundingBox]
    # Original frame dimensions
    frame_width: int
    frame_height: int
    # Total number of detections
    count: int


# COCO class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class ObstacleDetector:
    """
    Detects obstacles using YOLOv8-Nano object detection model.
    Outputs bounding boxes for relevant obstacle classes.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize obstacle detector.
        
        Args:
            config: Model configuration
        """
        self.logger = get_logger("obstacle_detector")
        self.config = config
        
        self.input_size = config.yolo_input_size
        self.conf_threshold = config.yolo_confidence_threshold
        self.nms_threshold = config.yolo_nms_threshold
        self.obstacle_classes = set(config.obstacle_classes)
        
        # 优先加载 PyTorch 模型
        self.pytorch_model = None
        try:
            self.logger.info("Loading PyTorch obstacle model (priority)...")
            self.pytorch_model = self._load_pytorch_model()
            if self.pytorch_model is not None:
                self.logger.info("✓ PyTorch obstacle model loaded successfully")
            else:
                self.logger.warning("✗ PyTorch model failed to load")
        except Exception as e:
            self.logger.warning(f"✗ Failed to load PyTorch model: {e}")
        
        # 若 PyTorch 加载失败，尝试 TFLite 作为备用
        self.model = None
        self.use_cv_fallback = True
        if self.pytorch_model is None:
            self.logger.info("PyTorch unavailable, falling back to TFLite...")
            loader = get_model_loader()
            self.model = loader.load_tflite(
                config.yolo_model_path,
                use_gpu=config.use_gpu_delegate,
                num_threads=config.num_threads
            )
            self.use_cv_fallback = isinstance(self.model, DummyModel)
        
        if self.use_cv_fallback and self.pytorch_model is None:
            self.logger.info("ML model missing, using CV fallback for obstacle detection")
        
        if self.pytorch_model or self.model:
            self.logger.info("Obstacle detector initialized")
        else:
            self.logger.error("Failed to load obstacle detection model")
    
    def _load_pytorch_model(self):
        """加载 PyTorch 障碍物检测模型"""
        try:
            # 优先加载合并训练的新模型，兜底用旧模型
            import os
            model_path = "training_checkpoints/obstacle_merged/weights/best.pt"
            if not os.path.exists(model_path):
                model_path = "training_checkpoints/obstacle/best.pt"
            self.logger.info(f"障碍物模型路径: {model_path}")
            detector = PyTorchObstacleDetector(
                model_path=model_path,
                conf_threshold=self.conf_threshold
            )
            return detector
        except Exception as e:
            self.logger.error(f"加载 PyTorch 模型失败: {e}")
            return None
    
    def detect(self, frame: np.ndarray) -> Optional[ObstacleDetectionResult]:
        """
        Detect obstacles in a frame.
        
        Args:
            frame: BGR image from camera (HWC format)
            
        Returns:
            ObstacleDetectionResult with detected obstacles, or None on error
        """
        try:
            h, w = frame.shape[:2]
            
            # Try PyTorch model first
            if self.pytorch_model is not None and self.pytorch_model.model is not None:
                detections = self.pytorch_model.detect(frame)
                obstacles = []
                for det in detections:
                    if det.class_id in self.obstacle_classes:
                        obstacles.append(BoundingBox(
                            x=det.x,
                            y=det.y,
                            width=det.width,
                            height=det.height,
                            confidence=det.confidence,
                            class_id=det.class_id,
                            class_name=det.class_name
                        ))
                return ObstacleDetectionResult(
                    obstacles=obstacles,
                    frame_width=w,
                    frame_height=h,
                    count=len(obstacles)
                )
            
            # Fall back to TFLite model
            if self.model is not None and not self.use_cv_fallback:
                input_tensor, scale, pad = self._preprocess(frame)
                output = self.model.predict(input_tensor)
                obstacles = self._postprocess(output, (w, h), scale, pad)
                return ObstacleDetectionResult(
                    obstacles=obstacles,
                    frame_width=w,
                    frame_height=h,
                    count=len(obstacles)
                )
            
            # Final fallback to CV
            return self._detect_cv(frame)
            
        except Exception as e:
            self.logger.error(f"Obstacle detection error: {e}")
            return None

    def _detect_cv(self, frame: np.ndarray) -> ObstacleDetectionResult:
        """Improved CV-based fallback for obstacle detection."""
        import cv2
        h, w = frame.shape[:2]
        
        # 1. Grayscale and noise reduction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Vertical Edge Detection (Sobel X)
        # Most obstacles (people, poles, walls) have strong vertical components
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_8u = np.uint8(abs_sobelx)
        
        # Threshold to get vertical edges
        _, thresh_sobel = cv2.threshold(sobel_8u, 40, 255, cv2.THRESH_BINARY)
        
        # 3. Combine with Canny for better contours
        edges = cv2.Canny(blurred, 30, 100)
        combined_edges = cv2.bitwise_or(thresh_sobel, edges)
        
        # 4. Filter contours by shape and density
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(combined_edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1200 < area < (w * h * 0.4):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = ch / cw
                
                # Heuristics: 
                # - Obstacles are usually taller than they are wide (people/poles)
                # - Or they are large blocks (furniture)
                # - Filter out very flat blobs (road markings)
                if aspect_ratio > 0.5 and (y + ch) > h * 0.3:
                    obstacles.append(BoundingBox(
                        x=x, y=y, width=cw, height=ch,
                        confidence=0.7,
                        class_id=0,
                        class_name="obstacle"
                    ))
        
        obstacles = self._nms(obstacles)
        return ObstacleDetectionResult(obstacles=obstacles, frame_width=w, frame_height=h, count=len(obstacles))
    
    def _preprocess(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for YOLO input.
        
        Args:
            frame: BGR image (HWC)
            
        Returns:
            Tuple of (tensor, scale, padding)
        """
        import cv2
        
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        
        # Calculate scale to fit within target size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded image (letterboxing)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        tensor = np.expand_dims(normalized, axis=0)
        
        return tensor, scale, (pad_x, pad_y)
    
    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int]
    ) -> List[BoundingBox]:
        """
        Post-process YOLO output to get bounding boxes.
        
        Args:
            output: Model output tensor
            original_size: (width, height) of original frame
            scale: Scale factor used in preprocessing
            pad: Padding (pad_x, pad_y) used in preprocessing
            
        Returns:
            List of BoundingBox objects for detected obstacles
        """
        # Handle YOLOv8 output format
        # Output shape: (1, num_detections, 85) or (1, 85, num_detections)
        if len(output.shape) == 3:
            if output.shape[1] == 85:
                # Transpose to (1, num_detections, 85)
                output = np.transpose(output, (0, 2, 1))
            predictions = output[0]  # Remove batch dimension
        else:
            predictions = output
        
        orig_w, orig_h = original_size
        pad_x, pad_y = pad
        
        boxes = []
        
        for pred in predictions:
            # Extract values based on YOLOv8 format
            # [x_center, y_center, width, height, class_scores...]
            if len(pred) >= 5:
                x_center, y_center, w, h = pred[:4]
                class_scores = pred[4:]
                
                # Get best class and confidence
                if len(class_scores) > 0:
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
                else:
                    continue
                
                # Filter by confidence
                if confidence < self.conf_threshold:
                    continue
                
                # Filter by obstacle classes
                if class_id not in self.obstacle_classes:
                    continue
                
                # Convert to pixel coordinates
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
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)
                
                # Get class name
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                
                boxes.append(BoundingBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        # Apply NMS
        boxes = self._nms(boxes)
        
        return boxes
    
    def _nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: List of detected bounding boxes
            
        Returns:
            Filtered list of boxes after NMS
        """
        if not boxes:
            return []
        
        # Sort by confidence (descending)
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            
            # Filter remaining boxes
            boxes = [
                box for box in boxes
                if self._iou(best, box) < self.nms_threshold
            ]
        
        return keep
    
    def _iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union between two boxes."""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        union = box1.area + box2.area - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def release(self):
        """Release model resources."""
        if self.model:
            self.model.release()
            self.model = None
