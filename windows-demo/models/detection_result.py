"""
Shared DetectionResult dataclass to avoid circular imports.
"""
from dataclasses import dataclass
import numpy as np
from typing import List
from models.obstacle_detector import BoundingBox


@dataclass
class DetectionResult:
    """Combined detection result from all models."""
    # Water detection
    water_mask: np.ndarray
    water_confidence: np.ndarray
    water_coverage: float
    
    # Obstacles
    obstacles: list  # List[BoundingBox]
    
    # Frame info
    frame_width: int
    frame_height: int
    
    # Hazard grid for path planning
    hazard_grid: np.ndarray
