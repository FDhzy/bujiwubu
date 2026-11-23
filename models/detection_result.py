"""
Shared DetectionResult dataclass to avoid circular imports.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional
from models.obstacle_detector import BoundingBox

# Import lazily to avoid circular deps at module load time
# TrafficLightResult is resolved at runtime via TYPE_CHECKING or Optional[Any]


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

    # Traffic light detection result (None if not detected / disabled)
    traffic_light: Optional[object] = field(default=None)
