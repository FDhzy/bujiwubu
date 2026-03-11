"""
Distance estimation module using computer vision.
Estimates real-world distances to puddles and obstacles.
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

from utils.logger import get_logger
from utils.config import CameraConfig

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class PuddleInfo:
    """Information about a detected puddle with distance estimates."""
    # Center position in frame (x, y)
    center_x: int
    center_y: int
    # Bounding box (x, y, width, height)
    bbox: Tuple[int, int, int, int]
    # Estimated distance in meters
    distance_m: float
    # Estimated diameter in meters
    diameter_m: float
    # Direction from user perspective
    direction: str  # "left", "center", "right"
    # Angle from center in degrees
    angle_degrees: float
    # Coverage ratio in frame
    coverage: float


@dataclass
class ObstacleInfo:
    """Information about a detected obstacle with distance estimates."""
    # Center position
    center_x: int
    center_y: int
    # Bounding box
    bbox: Tuple[int, int, int, int]
    # Estimated distance in meters
    distance_m: float
    # Estimated width in meters
    width_m: float
    # Direction
    direction: str
    # Object class name
    class_name: str
    # Confidence
    confidence: float


@dataclass
class NavigationGuidance:
    """Precise navigation guidance with step-by-step instructions."""
    # Primary instruction text
    instruction: str
    # Detailed description
    description: str
    # Steps forward before action
    steps_forward: int
    # Steps to move sideways (positive=right, negative=left)
    steps_sideways: int
    # Direction to move ("left", "right", "straight", "stop")
    move_direction: str
    # Distance to nearest hazard in meters
    hazard_distance_m: float
    # Hazard size in meters
    hazard_size_m: float
    # Safety level
    safety_level: str  # "safe", "caution", "danger"


class DistanceEstimator:
    """
    Estimates real-world distances using computer vision.
    Uses perspective geometry and reference sizes for estimation.
    """
    
    # Average step length in meters (2 steps = 1 meter)
    STEP_LENGTH_M = 0.5
    
    # Reference heights for distance estimation (meters)
    REFERENCE_HEIGHTS = {
        "person": 1.7,
        "car": 1.5,
        "bicycle": 1.0,
        "bench": 0.45,
        "fire hydrant": 0.6,
        "traffic light": 0.4,  # Just the light part
    }
    
    def __init__(self, camera_config: CameraConfig):
        """
        Initialize distance estimator.
        
        Args:
            camera_config: Camera configuration with resolution and FOV
        """
        self.logger = get_logger("distance")
        self.config = camera_config
        
        # Camera parameters
        self.frame_width = camera_config.resolution[0]
        self.frame_height = camera_config.resolution[1]
        self.fov_h = camera_config.fov_horizontal
        
        # Calculate focal length (in pixels) from FOV
        self.focal_length = self.frame_width / (2 * math.tan(math.radians(self.fov_h / 2)))
        
        # Assumed camera height from ground (meters) - typical phone held at chest level
        self.camera_height_m = 1.2
        
        # Ground plane parameters for perspective
        # Horizon line is typically at vertical center or slightly above
        self.horizon_y = self.frame_height * 0.4
        
        # Precise camera geometry
        # Assumed camera pitch (tilt) in degrees - looking 30 degrees down
        self.camera_pitch_deg = 15.0
        
        # Distance tracking for temporal smoothing
        self._distance_cache = {}  # {hazard_id: smoothed_distance}
        self._alpha = 0.3  # Smoothing factor (0.1 to 0.5)
        
        self.logger.info(
            f"DistanceEstimator initialized: {self.frame_width}x{self.frame_height}, "
            f"FOV={self.fov_h}deg, focal={self.focal_length:.1f}px"
        )
    
    def estimate_puddle_info(
        self,
        water_mask: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> List[PuddleInfo]:
        """
        Analyze water mask to get puddle information with distances.
        
        Args:
            water_mask: Binary mask of water regions
            frame_shape: (height, width) of frame
            
        Returns:
            List of PuddleInfo for each detected puddle
        """
        if not CV2_AVAILABLE or water_mask is None:
            return []
        
        if not np.any(water_mask > 0):
            return []
        
        frame_h, frame_w = frame_shape[:2]
        puddles = []
        
        # Find connected components (individual puddles)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            water_mask.astype(np.uint8)
        )
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:  # Skip tiny regions
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            center_x = int(centroids[i, 0])
            center_y = int(centroids[i, 1])
            bottom_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            
            # Estimate distance based on bottom-most vertical position
            # Use bottom_y instead of center_y for the entry point of the puddle
            raw_distance = self._estimate_ground_distance(bottom_y, frame_h)
            
            # Apply temporal smoothing (hazard_id for puddles can be their initial center)
            hazard_id = f"puddle_{i}"
            distance = self._smooth_distance(hazard_id, raw_distance)
            
            # Estimate physical size (diameter)
            diameter = self._estimate_ground_size(w, h, bottom_y, frame_h)
            
            # Calculate direction
            direction, angle = self._get_direction(center_x, frame_w)
            
            # Coverage ratio
            coverage = area / (frame_w * frame_h)
            
            puddles.append(PuddleInfo(
                center_x=center_x,
                center_y=center_y,
                bbox=(x, y, w, h),
                distance_m=distance,
                diameter_m=diameter,
                direction=direction,
                angle_degrees=angle,
                coverage=coverage
            ))
        
        # Sort by distance (closest first)
        puddles.sort(key=lambda p: p.distance_m)
        
        return puddles
    
    def estimate_obstacle_info(
        self,
        obstacles: list,
        frame_shape: Tuple[int, int]
    ) -> List[ObstacleInfo]:
        """
        Estimate distances to obstacles.
        
        Args:
            obstacles: List of BoundingBox objects
            frame_shape: (height, width) of frame
            
        Returns:
            List of ObstacleInfo with distance estimates
        """
        if not obstacles:
            return []
        
        frame_h, frame_w = frame_shape[:2]
        result = []
        
        for obs in obstacles:
            x, y, w, h = obs.bbox
            center_x = x + w // 2
            center_y = y + h // 2
            bottom_y = y + h
            
            # Estimate distance using object size or ground contact point
            if obs.class_name in self.REFERENCE_HEIGHTS:
                # Use known object height for distance estimation
                ref_height = self.REFERENCE_HEIGHTS[obs.class_name]
                dist_h = self._estimate_distance_from_height(h, ref_height)
                dist_g = self._estimate_ground_distance(bottom_y, frame_h)
                # Weighted fusion: Height is usually more accurate if fully visible
                raw_distance = 0.6 * dist_h + 0.4 * dist_g
            else:
                # Use ground contact point
                raw_distance = self._estimate_ground_distance(bottom_y, frame_h)
            
            # Smooth distance
            hazard_id = f"obs_{obs.class_name}_{center_x // 50}" # Simple spatial clustering
            distance = self._smooth_distance(hazard_id, raw_distance)
            
            # Estimate width
            width_m = self._estimate_width_at_distance(w, distance)
            
            # Direction
            direction, _ = self._get_direction(center_x, frame_w)
            
            result.append(ObstacleInfo(
                center_x=center_x,
                center_y=center_y,
                bbox=obs.bbox,
                distance_m=distance,
                width_m=width_m,
                direction=direction,
                class_name=obs.class_name,
                confidence=obs.confidence
            ))
        
        # Sort by distance
        result.sort(key=lambda o: o.distance_m)
        
        return result
    
    def generate_guidance(
        self,
        puddles: List[PuddleInfo],
        obstacles: List[ObstacleInfo]
    ) -> NavigationGuidance:
        """
        Generate precise navigation guidance based on detected hazards.
        
        Args:
            puddles: List of detected puddles with distances
            obstacles: List of detected obstacles with distances
            
        Returns:
            NavigationGuidance with step-by-step instructions
        """
        # Find closest hazard
        closest_puddle = puddles[0] if puddles else None
        closest_obstacle = obstacles[0] if obstacles else None
        
        # No hazards - clear path
        if not closest_puddle and not closest_obstacle:
            return NavigationGuidance(
                instruction="clear",
                description="Path is clear ahead",
                steps_forward=10,
                steps_sideways=0,
                move_direction="straight",
                hazard_distance_m=float('inf'),
                hazard_size_m=0,
                safety_level="safe"
            )
        
        # Determine primary hazard
        if closest_puddle and closest_obstacle:
            if closest_puddle.distance_m < closest_obstacle.distance_m:
                return self._guidance_for_puddle(closest_puddle, closest_obstacle)
            else:
                return self._guidance_for_obstacle(closest_obstacle, closest_puddle)
        elif closest_puddle:
            return self._guidance_for_puddle(closest_puddle, None)
        else:
            return self._guidance_for_obstacle(closest_obstacle, None)
    
    def _guidance_for_puddle(
        self,
        puddle: PuddleInfo,
        obstacle: Optional[ObstacleInfo]
    ) -> NavigationGuidance:
        """Generate guidance for puddle avoidance."""
        distance = puddle.distance_m
        diameter = puddle.diameter_m
        
        # Calculate steps
        steps_forward = max(0, int((distance - 0.5) / self.STEP_LENGTH_M))
        
        # Determine avoidance direction
        if puddle.direction == "center":
            # Puddle is centered - check which side has more space
            if obstacle and obstacle.direction == "left":
                move_dir = "right"
                steps_side = max(1, int((diameter / 2 + 0.3) / self.STEP_LENGTH_M))
            elif obstacle and obstacle.direction == "right":
                move_dir = "left"
                steps_side = -max(1, int((diameter / 2 + 0.3) / self.STEP_LENGTH_M))
            else:
                # Default to left
                move_dir = "left"
                steps_side = -max(1, int((diameter / 2 + 0.3) / self.STEP_LENGTH_M))
        elif puddle.direction == "left":
            move_dir = "right"
            steps_side = max(1, int((diameter / 2 + 0.3) / self.STEP_LENGTH_M))
        else:
            move_dir = "left"
            steps_side = -max(1, int((diameter / 2 + 0.3) / self.STEP_LENGTH_M))
        
        # Safety level
        if distance < 1.0:
            safety = "danger"
        elif distance < 2.0:
            safety = "caution"
        else:
            safety = "safe"
        
        # Generate description
        desc = (
            f"Puddle {distance:.1f} meters ahead, "
            f"approximately {diameter:.1f} meter{'s' if diameter > 1 else ''} wide"
        )
        
        return NavigationGuidance(
            instruction="puddle_precise",
            description=desc,
            steps_forward=steps_forward,
            steps_sideways=steps_side,
            move_direction=move_dir,
            hazard_distance_m=distance,
            hazard_size_m=diameter,
            safety_level=safety
        )
    
    def _guidance_for_obstacle(
        self,
        obstacle: ObstacleInfo,
        puddle: Optional[PuddleInfo]
    ) -> NavigationGuidance:
        """Generate guidance for obstacle avoidance."""
        distance = obstacle.distance_m
        width = obstacle.width_m
        
        steps_forward = max(0, int((distance - 0.5) / self.STEP_LENGTH_M))
        
        # Determine avoidance direction
        if obstacle.direction == "center":
            if puddle and puddle.direction == "left":
                move_dir = "right"
                steps_side = max(1, int((width / 2 + 0.5) / self.STEP_LENGTH_M))
            elif puddle and puddle.direction == "right":
                move_dir = "left"
                steps_side = -max(1, int((width / 2 + 0.5) / self.STEP_LENGTH_M))
            else:
                move_dir = "left"
                steps_side = -max(1, int((width / 2 + 0.5) / self.STEP_LENGTH_M))
        elif obstacle.direction == "left":
            move_dir = "right"
            steps_side = max(1, int((width / 2 + 0.5) / self.STEP_LENGTH_M))
        else:
            move_dir = "left"
            steps_side = -max(1, int((width / 2 + 0.5) / self.STEP_LENGTH_M))
        
        if distance < 1.5:
            safety = "danger"
        elif distance < 3.0:
            safety = "caution"
        else:
            safety = "safe"
        
        desc = f"{obstacle.class_name} {distance:.1f} meters ahead"
        
        return NavigationGuidance(
            instruction="obstacle_precise",
            description=desc,
            steps_forward=steps_forward,
            steps_sideways=steps_side,
            move_direction=move_dir,
            hazard_distance_m=distance,
            hazard_size_m=width,
            safety_level=safety
        )
    
    def _estimate_ground_distance(self, y_pixel: int, frame_h: int) -> float:
        """
        Estimate distance to a point on the ground plane using trigonometry.
        Improved with dynamic horizon adjustment based on frame analysis.
        """
        # Dynamic horizon estimation: Look for the 'limit of ground' 
        # based on where vertical gradients peak (usually buildings/walls meeting floor)
        horizon_y = self.horizon_y
        
        # Normalize y position (0 = horizon, 1 = bottom of frame)
        if y_pixel <= horizon_y:
            return 20.0
        
        # 1. Calculate vertical angle from optical axis (alpha)
        cy = frame_h / 2
        # Correct for vertical FOV
        v_fov_rad = math.radians(self.fov_h * (frame_h / self.frame_width))
        alpha_rad = math.atan((y_pixel - cy) * math.tan(v_fov_rad/2) / (frame_h/2))
        
        # 2. Camera pitch (tilt) in radians
        pitch_rad = math.radians(self.camera_pitch_deg)
        
        # 3. Total angle to ground point (depression angle)
        total_angle = pitch_rad + alpha_rad
        
        # 4. Calculate distance: D = H / tan(angle)
        # Avoid division by zero or negative angles
        if total_angle <= 0.02: 
            return 20.0
            
        distance = self.camera_height_m / math.tan(total_angle)
        
        # Scaling correction for typical mobile camera distortion
        # (Slightly increase distance for points further up)
        dist_factor = 1.0 + (1.0 - (y_pixel / frame_h)) * 0.2
        distance *= dist_factor

        # Clamp to reasonable range
        return max(0.2, min(distance, 20.0))

    def _smooth_distance(self, hazard_id: str, new_val: float) -> float:
        """Apply Exponential Moving Average for smoothing."""
        if hazard_id not in self._distance_cache:
            self._distance_cache[hazard_id] = new_val
            return new_val
            
        smoothed = self._alpha * new_val + (1 - self._alpha) * self._distance_cache[hazard_id]
        self._distance_cache[hazard_id] = smoothed
        
        # Cleanup cache if it grows too large
        if len(self._distance_cache) > 50:
            self._distance_cache.clear()
            
        return smoothed
    
    def _estimate_ground_size(
        self,
        pixel_w: int,
        pixel_h: int,
        y_pixel: int,
        frame_h: int
    ) -> float:
        """Estimate physical size of object on ground."""
        distance = self._estimate_ground_distance(y_pixel, frame_h)
        
        # Size in meters = (pixel_size * distance) / focal_length
        size_w = (pixel_w * distance) / self.focal_length
        size_h = (pixel_h * distance) / self.focal_length
        
        # Return average as diameter estimate
        return (size_w + size_h) / 2
    
    def _estimate_distance_from_height(
        self,
        pixel_height: int,
        real_height: float
    ) -> float:
        """Estimate distance using known object height."""
        if pixel_height <= 0:
            return 20.0
        
        # distance = (real_height * focal_length) / pixel_height
        distance = (real_height * self.focal_length) / pixel_height
        
        return max(0.5, min(distance, 20.0))
    
    def _estimate_width_at_distance(
        self,
        pixel_width: int,
        distance: float
    ) -> float:
        """Estimate physical width given pixel width and distance."""
        return (pixel_width * distance) / self.focal_length
    
    def _get_direction(
        self,
        x_pixel: int,
        frame_w: int
    ) -> Tuple[str, float]:
        """Get direction name and angle from horizontal position."""
        # Calculate angle from center
        center_x = frame_w / 2
        offset = x_pixel - center_x
        angle = math.degrees(math.atan2(offset, self.focal_length))
        
        # Determine direction
        if abs(angle) < 10:
            direction = "center"
        elif angle < 0:
            direction = "left"
        else:
            direction = "right"
        
        return direction, angle
