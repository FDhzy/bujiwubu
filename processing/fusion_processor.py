"""
Fusion processor for combining multiple detection results.
Merges water segmentation and obstacle detection into unified hazard map.
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from processing.segmentation_processor import SegmentationResult, PuddleRegion
from processing.detection_processor import ProcessedDetection
from models.obstacle_detector import BoundingBox


@dataclass
class HazardZone:
    """A zone in the frame with hazard information."""
    # Grid position (row, col)
    row: int
    col: int
    # Hazard scores
    puddle_coverage: float  # 0-1, fraction covered by water
    obstacle_coverage: float  # 0-1, fraction covered by obstacles
    combined_hazard: float  # Weighted combination
    # Whether this zone is safe to traverse
    is_safe: bool


@dataclass
class FusedDetectionResult:
    """Combined result from all detection methods."""
    # Water detection results
    water_mask: np.ndarray
    water_confidence: np.ndarray
    puddle_regions: List[PuddleRegion]
    water_coverage: float
    
    # Obstacle detection results
    obstacles: List[BoundingBox]
    obstacles_left: int
    obstacles_center: int
    obstacles_right: int
    
    # Hazard grid (for path planning)
    hazard_grid: np.ndarray  # Shape: (grid_rows, grid_cols)
    hazard_zones: List[List[HazardZone]]
    
    # Frame info
    frame_width: int
    frame_height: int
    
    # Overall safety assessment
    overall_safety: float  # 0-1, higher = safer


class FusionProcessor:
    """
    Fuses results from multiple detection methods into a unified hazard map.
    Combines water segmentation, obstacle detection, and traditional CV.
    """
    
    def __init__(
        self,
        grid_rows: int = 8,
        grid_cols: int = 8,
        puddle_weight: float = 0.4,
        obstacle_weight: float = 0.6,
        focus_ratio: float = 0.67
    ):
        """
        Initialize fusion processor.
        
        Args:
            grid_rows: Number of rows in hazard grid
            grid_cols: Number of columns in hazard grid
            puddle_weight: Weight for puddle hazards
            obstacle_weight: Weight for obstacle hazards
            focus_ratio: Focus on bottom portion of frame (ground level)
        """
        self.logger = get_logger("fusion")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.puddle_weight = puddle_weight
        self.obstacle_weight = obstacle_weight
        self.focus_ratio = focus_ratio
        
        # Temporal smoothing state
        self._prev_hazard_grid = None
        self._smoothing_factor = 0.6  # Weight for new data (0-1)
    
    def fuse(
        self,
        water_result: Optional[SegmentationResult],
        obstacle_result: Optional[ProcessedDetection],
        reflection_mask: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int, int] = (480, 640, 3)
    ) -> FusedDetectionResult:
        """
        Fuse multiple detection results.
        
        Args:
            water_result: Result from water detection model
            obstacle_result: Result from obstacle detection model
            reflection_mask: Optional mask from traditional CV
            frame_shape: Shape of original frame (H, W, C)
            
        Returns:
            FusedDetectionResult with combined hazard map
        """
        frame_h, frame_w = frame_shape[:2]
        
        # Get water mask (combine ML and traditional CV if available)
        if water_result is not None:
            water_mask = water_result.binary_mask
            water_confidence = water_result.confidence_map
            puddle_regions = water_result.regions
            water_coverage = water_result.coverage
        else:
            water_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            water_confidence = np.zeros((frame_h, frame_w), dtype=np.float32)
            puddle_regions = []
            water_coverage = 0.0
        
        # Combine with reflection mask if available
        if reflection_mask is not None:
            # Resize if needed
            if reflection_mask.shape != water_mask.shape:
                import cv2
                reflection_mask = cv2.resize(
                    reflection_mask.astype(np.float32),
                    (frame_w, frame_h)
                ).astype(np.uint8)
            
            # Union of both masks
            water_mask = np.maximum(water_mask, reflection_mask)
        
        # Get obstacle info
        if obstacle_result is not None:
            obstacles = obstacle_result.obstacles
            obstacles_left = obstacle_result.obstacles_left
            obstacles_center = obstacle_result.obstacles_center
            obstacles_right = obstacle_result.obstacles_right
        else:
            obstacles = []
            obstacles_left = 0
            obstacles_center = 0
            obstacles_right = 0
        
        # Create obstacle mask from bounding boxes
        obstacle_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for obs in obstacles:
            x, y, w, h = obs.bbox
            obstacle_mask[y:y+h, x:x+w] = 1
        
        # Calculate hazard grid
        raw_hazard_grid, hazard_zones = self._compute_hazard_grid(
            water_mask, obstacle_mask, frame_h, frame_w
        )
        
        # Apply temporal smoothing
        if self._prev_hazard_grid is None or self._prev_hazard_grid.shape != raw_hazard_grid.shape:
            hazard_grid = raw_hazard_grid
        else:
            # Simple IIR filter: smooth = alpha * new + (1 - alpha) * prev
            hazard_grid = (self._smoothing_factor * raw_hazard_grid + 
                           (1 - self._smoothing_factor) * self._prev_hazard_grid)
        
        self._prev_hazard_grid = hazard_grid.copy()
        
        # Calculate overall safety
        overall_safety = self._compute_overall_safety(hazard_grid)
        
        return FusedDetectionResult(
            water_mask=water_mask,
            water_confidence=water_confidence,
            puddle_regions=puddle_regions,
            water_coverage=water_coverage,
            obstacles=obstacles,
            obstacles_left=obstacles_left,
            obstacles_center=obstacles_center,
            obstacles_right=obstacles_right,
            hazard_grid=hazard_grid,
            hazard_zones=hazard_zones,
            frame_width=frame_w,
            frame_height=frame_h,
            overall_safety=overall_safety
        )
    
    def _compute_hazard_grid(
        self,
        water_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        frame_h: int,
        frame_w: int
    ) -> Tuple[np.ndarray, List[List[HazardZone]]]:
        """Compute hazard grid from masks."""
        # Focus on bottom portion of frame
        focus_start = int(frame_h * (1 - self.focus_ratio))
        
        # Calculate cell dimensions
        focus_h = frame_h - focus_start
        cell_h = focus_h // self.grid_rows
        cell_w = frame_w // self.grid_cols
        
        hazard_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        hazard_zones = []
        
        for row in range(self.grid_rows):
            zone_row = []
            y_start = focus_start + row * cell_h
            y_end = y_start + cell_h
            
            for col in range(self.grid_cols):
                x_start = col * cell_w
                x_end = x_start + cell_w
                
                # Extract cell regions from masks
                water_cell = water_mask[y_start:y_end, x_start:x_end]
                obstacle_cell = obstacle_mask[y_start:y_end, x_start:x_end]
                
                # Calculate coverage ratios
                cell_size = water_cell.size
                puddle_coverage = np.sum(water_cell) / cell_size if cell_size > 0 else 0
                obstacle_coverage = np.sum(obstacle_cell) / cell_size if cell_size > 0 else 0
                
                # Combined hazard (weighted)
                combined = (
                    self.puddle_weight * puddle_coverage +
                    self.obstacle_weight * obstacle_coverage
                )
                
                # Determine if safe
                is_safe = combined < 0.3 and obstacle_coverage < 0.2
                
                zone = HazardZone(
                    row=row,
                    col=col,
                    puddle_coverage=puddle_coverage,
                    obstacle_coverage=obstacle_coverage,
                    combined_hazard=combined,
                    is_safe=is_safe
                )
                
                zone_row.append(zone)
                hazard_grid[row, col] = combined
            
            hazard_zones.append(zone_row)
        
        return hazard_grid, hazard_zones
    
    def _compute_overall_safety(self, hazard_grid: np.ndarray) -> float:
        """Compute overall safety score from hazard grid."""
        # Focus on center and bottom rows (user's likely path)
        center_cols = self.grid_cols // 4
        center_region = hazard_grid[:, center_cols:-center_cols]
        
        # Higher hazard = lower safety
        mean_hazard = np.mean(center_region)
        safety = 1.0 - min(mean_hazard, 1.0)
        
        return float(safety)
