"""
Segmentation output processor for water detection.
Post-processes DeepLabV3 outputs to get clean water masks.
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from utils.logger import get_logger


@dataclass
class PuddleRegion:
    """Information about a detected puddle region."""
    # Bounding box (x, y, width, height)
    bbox: Tuple[int, int, int, int]
    # Center point (x, y)
    center: Tuple[int, int]
    # Area in pixels
    area: int
    # Mean confidence score
    confidence: float
    # Position in frame (left/center/right)
    position: str


@dataclass
class SegmentationResult:
    """Processed segmentation output."""
    # Binary mask (1 = water)
    binary_mask: np.ndarray
    # Confidence map (0-1)
    confidence_map: np.ndarray
    # List of distinct puddle regions
    regions: List[PuddleRegion]
    # Total water coverage ratio
    coverage: float


class SegmentationProcessor:
    """
    Processes semantic segmentation output from DeepLabV3.
    Converts raw model output to clean binary masks and region info.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_region_area: int = 500,
        water_class_index: int = 0
    ):
        """
        Initialize segmentation processor.
        
        Args:
            confidence_threshold: Threshold for binary mask
            min_region_area: Minimum pixels for valid puddle region
            water_class_index: Index of water class in model output
        """
        self.logger = get_logger("segmentation")
        self.threshold = confidence_threshold
        self.min_area = min_region_area
        self.water_class = water_class_index
    
    def process(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> SegmentationResult:
        """
        Process segmentation model output.
        
        Args:
            output: Model output tensor
            original_size: (height, width) of original frame
            
        Returns:
            SegmentationResult with mask and region info
        """
        import cv2
        
        # Extract water channel from output
        confidence_map = self._extract_water_channel(output)
        
        # Resize to original frame size
        confidence_map = cv2.resize(
            confidence_map,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create binary mask
        binary_mask = (confidence_map > self.threshold).astype(np.uint8)
        
        # Clean up mask
        binary_mask = self._clean_mask(binary_mask)
        
        # Find puddle regions
        regions = self._find_regions(binary_mask, confidence_map, original_size)
        
        # Calculate coverage
        coverage = np.sum(binary_mask) / binary_mask.size
        
        return SegmentationResult(
            binary_mask=binary_mask,
            confidence_map=confidence_map,
            regions=regions,
            coverage=coverage
        )
    
    def _extract_water_channel(self, output: np.ndarray) -> np.ndarray:
        """Extract water probability map from model output."""
        # Handle different output formats
        if len(output.shape) == 4:
            # NHWC or NCHW format
            if output.shape[-1] > 1:
                # Multi-class: extract water channel
                confidence = output[0, :, :, self.water_class]
            else:
                # Single channel
                confidence = output[0, :, :, 0]
        elif len(output.shape) == 3:
            if output.shape[0] == 1:
                # NHW format
                confidence = output[0]
            else:
                confidence = output[:, :, self.water_class] if output.shape[-1] > 1 else output[:, :, 0]
        else:
            confidence = output.squeeze()
        
        # Apply sigmoid if logits
        if confidence.min() < 0 or confidence.max() > 1:
            confidence = 1 / (1 + np.exp(-confidence))
        
        return confidence.astype(np.float32)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean mask."""
        import cv2
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _find_regions(
        self,
        mask: np.ndarray,
        confidence: np.ndarray,
        frame_size: Tuple[int, int]
    ) -> List[PuddleRegion]:
        """Find distinct puddle regions in mask."""
        import cv2
        
        regions = []
        frame_h, frame_w = frame_size
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < self.min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            center_x = int(centroids[i, 0])
            center_y = int(centroids[i, 1])
            
            # Calculate mean confidence for this region
            region_mask = labels == i
            mean_conf = float(np.mean(confidence[region_mask]))
            
            # Determine position in frame
            if center_x < frame_w / 3:
                position = "left"
            elif center_x > 2 * frame_w / 3:
                position = "right"
            else:
                position = "center"
            
            regions.append(PuddleRegion(
                bbox=(x, y, w, h),
                center=(center_x, center_y),
                area=area,
                confidence=mean_conf,
                position=position
            ))
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r.area, reverse=True)
        
        return regions
