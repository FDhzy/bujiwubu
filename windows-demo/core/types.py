"""
Unified data types and structures for the application.
Consolidates various dataclasses and enums to reduce redundancy.
"""
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Optional, Dict
import numpy as np
import time

# --- Enums ---

class AppState(Enum):
    """Application state levels."""
    IDLE = "idle"
    SHOE_SETUP = "shoe_setup"
    STARTING = "starting"
    ACTIVE = "active"
    PROCESSING = "processing"
    WARNING = "warning"
    PAUSED = "paused"
    ERROR = "error"

class ShoeType(Enum):
    """Types of shoes with different water resistance."""
    UNKNOWN = "unknown"
    SANDALS = "sandals"
    LOW_SHOES = "low_shoes"
    ANKLE_SHOES = "ankle_shoes"
    BOOTS = "boots"
    HIGH_BOOTS = "high_boots"

class AudioPriority(IntEnum):
    """Audio message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# --- Dataclasses ---

@dataclass
class BoundingBox:
    """Detection bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

@dataclass
class HazardInfo:
    """Base class for detected hazards (puddles or obstacles)."""
    center_x: int
    center_y: int
    bbox: Tuple[int, int, int, int]
    distance_m: float
    direction: str  # "left", "center", "right"
    angle_degrees: float
    confidence: float

@dataclass
class PuddleInfo(HazardInfo):
    """Information about a detected puddle."""
    diameter_m: float
    coverage: float

@dataclass
class ObstacleInfo(HazardInfo):
    """Information about a detected obstacle."""
    width_m: float
    class_name: str

@dataclass
class NavigationGuidance:
    """Navigation instruction result."""
    instruction: str
    description: str
    steps_forward: int
    steps_sideways: int
    move_direction: str
    hazard_distance_m: float
    hazard_size_m: float
    safety_level: str  # "safe", "caution", "danger"

@dataclass
class HazardZone:
    """A zone in the frame with hazard information."""
    row: int
    col: int
    puddle_coverage: float
    obstacle_coverage: float
    combined_hazard: float
    is_safe: bool

@dataclass
class DetectionResult:
    """Container for processed detection results."""
    obstacles: List[BoundingBox]
    frame_width: int
    frame_height: int
    obstacles_left: int
    obstacles_center: int
    obstacles_right: int
    closest_obstacle: Optional[BoundingBox]
    closest_distance: float  # Relative to frame height

@dataclass
class PreciseInstruction:
    """Precise navigation instruction for TTS."""
    text_en: str
    text_zh: str
    priority: int  # 1=low, 2=normal, 3=high, 4=critical
    instruction_type: str  # "puddle", "obstacle", "clear", "stop"

@dataclass
class ShoeAnalysisResult:
    """Result of shoe analysis."""
    shoe_type: ShoeType
    confidence: float
    puddle_warning_threshold: float
    puddle_critical_threshold: float
    safe_puddle_depth_cm: float
    description: str
    image_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class FusedResult:
    """Unified result from all detection methods."""
    water_mask: np.ndarray
    puddles: List[PuddleInfo]
    obstacles: List[ObstacleInfo]
    hazard_grid: np.ndarray
    overall_safety: float
    frame_size: Tuple[int, int]
