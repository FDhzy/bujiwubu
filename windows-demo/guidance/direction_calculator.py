"""
Direction calculator for converting detection coordinates to navigation directions.
Maps pixel positions to real-world directions for user guidance.
"""
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass

from utils.logger import get_logger
from utils.config import CameraConfig


@dataclass
class DirectionInfo:
    """Information about a direction/location."""
    # Angle from center (-90 to 90, negative = left)
    angle: float
    # Distance estimate (0 = near, 1 = far)
    distance: float
    # Direction name (left, center, right)
    zone: str
    # More detailed direction
    detailed_direction: str


class DirectionCalculator:
    """
    Calculates real-world directions from image coordinates.
    Uses camera field-of-view to estimate angles.
    """
    
    def __init__(self, camera_config: CameraConfig):
        """
        Initialize direction calculator.
        
        Args:
            camera_config: Camera configuration with FOV info
        """
        self.logger = get_logger("direction")
        
        self.fov_horizontal = camera_config.fov_horizontal
        self.frame_width = camera_config.resolution[0]
        self.frame_height = camera_config.resolution[1]
        
        # Precompute pixels per degree
        self.pixels_per_degree = self.frame_width / self.fov_horizontal
        
        self.logger.info(
            f"DirectionCalculator initialized: FOV={self.fov_horizontal}deg, "
            f"resolution={self.frame_width}x{self.frame_height}"
        )
    
    def pixel_to_angle(self, x: int) -> float:
        """
        Convert horizontal pixel position to angle from center.
        
        Args:
            x: Pixel x-coordinate
            
        Returns:
            Angle in degrees (negative = left, positive = right)
        """
        center_x = self.frame_width / 2
        offset = x - center_x
        angle = offset / self.pixels_per_degree
        return angle
    
    def pixel_to_direction(self, x: int, y: int) -> DirectionInfo:
        """
        Convert pixel coordinates to direction information.
        
        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            
        Returns:
            DirectionInfo with angle, distance, and zone
        """
        # Calculate horizontal angle
        angle = self.pixel_to_angle(x)
        
        # Estimate distance from vertical position
        # Bottom of frame = near (0), top = far (1)
        distance = 1.0 - (y / self.frame_height)
        
        # Determine zone
        zone = self._angle_to_zone(angle)
        
        # Get detailed direction
        detailed = self._get_detailed_direction(angle, distance)
        
        return DirectionInfo(
            angle=angle,
            distance=distance,
            zone=zone,
            detailed_direction=detailed
        )
    
    def _angle_to_zone(self, angle: float) -> str:
        """Convert angle to zone name."""
        if angle < -15:
            return "left"
        elif angle > 15:
            return "right"
        else:
            return "center"
    
    def _get_detailed_direction(self, angle: float, distance: float) -> str:
        """Get detailed direction description."""
        # Horizontal direction
        if abs(angle) < 5:
            h_dir = "straight ahead"
        elif abs(angle) < 15:
            h_dir = "slightly left" if angle < 0 else "slightly right"
        elif abs(angle) < 30:
            h_dir = "left" if angle < 0 else "right"
        else:
            h_dir = "far left" if angle < 0 else "far right"
        
        # Distance modifier
        if distance < 0.3:
            d_mod = "very close"
        elif distance < 0.5:
            d_mod = "close"
        elif distance < 0.7:
            d_mod = "ahead"
        else:
            d_mod = "far"
        
        return f"{d_mod}, {h_dir}"
    
    def bbox_to_direction(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> DirectionInfo:
        """
        Convert bounding box to direction information.
        
        Args:
            bbox: (x, y, width, height) of bounding box
            
        Returns:
            DirectionInfo for center of bbox
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        return self.pixel_to_direction(center_x, center_y)
    
    def estimate_distance_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        known_height_m: float = 1.7
    ) -> float:
        """
        Estimate real distance to object using bounding box height.
        
        Args:
            bbox: (x, y, width, height) of bounding box
            known_height_m: Known real-world height of object in meters
            
        Returns:
            Estimated distance in meters (approximate)
        """
        _, _, _, box_height = bbox
        
        if box_height <= 0:
            return float('inf')
        
        # Simple pinhole camera model approximation
        # distance = (known_height * focal_length) / pixel_height
        # Approximate focal length from FOV
        fov_rad = math.radians(self.fov_horizontal)
        focal_length = self.frame_width / (2 * math.tan(fov_rad / 2))
        
        # Estimate using vertical FOV (assuming square pixels)
        fov_vertical = 2 * math.atan(
            self.frame_height / (2 * focal_length)
        )
        focal_v = self.frame_height / (2 * math.tan(fov_vertical / 2))
        
        distance = (known_height_m * focal_v) / box_height
        
        return distance
    
    def get_turn_recommendation(
        self,
        hazard_positions: List[Tuple[int, int]]
    ) -> Tuple[float, str]:
        """
        Calculate recommended turn direction based on hazard positions.
        
        Args:
            hazard_positions: List of (x, y) positions of hazards
            
        Returns:
            Tuple of (recommended_angle, direction_name)
        """
        if not hazard_positions:
            return 0.0, "straight"
        
        # Calculate weighted average position of hazards
        # Weight by inverse distance (closer = more important)
        total_weight = 0.0
        weighted_x = 0.0
        
        for x, y in hazard_positions:
            distance = 1.0 - (y / self.frame_height)
            weight = 1.0 / (distance + 0.1)  # Avoid division by zero
            
            weighted_x += x * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_hazard_x = weighted_x / total_weight
        else:
            return 0.0, "straight"
        
        # Turn away from hazards
        hazard_angle = self.pixel_to_angle(int(avg_hazard_x))
        
        # Recommend turning opposite direction
        if abs(hazard_angle) < 10:
            # Hazard is center, check which side has more space
            left_hazards = sum(1 for x, _ in hazard_positions if x < self.frame_width / 2)
            right_hazards = len(hazard_positions) - left_hazards
            
            if left_hazards > right_hazards:
                return 20.0, "right"
            elif right_hazards > left_hazards:
                return -20.0, "left"
            else:
                return 0.0, "stop"
        elif hazard_angle < 0:
            # Hazard on left, turn right
            return min(30.0, abs(hazard_angle) + 10), "right"
        else:
            # Hazard on right, turn left
            return -min(30.0, hazard_angle + 10), "left"
    
    def quantize_direction(self, angle: float) -> str:
        """
        Quantize angle to discrete direction name.
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Direction name (straight, slight_left, etc.)
        """
        abs_angle = abs(angle)
        
        if abs_angle < 7.5:
            return "straight"
        elif abs_angle < 22.5:
            return "slight_left" if angle < 0 else "slight_right"
        elif abs_angle < 45:
            return "left" if angle < 0 else "right"
        else:
            return "hard_left" if angle < 0 else "hard_right"
