"""
Path planner using A* algorithm for safe navigation.
Plans walking routes that avoid puddles and obstacles.
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import heapq

from utils.logger import get_logger
from utils.config import PathPlanningConfig, get_config
from models.obstacle_detector import BoundingBox


@dataclass
class NavigationResult:
    """Result from path planning."""
    # Recommended turn angle (negative = left, positive = right)
    turn_angle: float
    # Direction name (e.g., "slight_left", "straight")
    direction: str
    # Safety score (0-1, higher = safer)
    safety_score: float
    # Distance to nearest hazard (relative, 0-1)
    hazard_distance: float
    # Whether immediate stop is recommended
    should_stop: bool
    # Planned path as list of (row, col) grid positions
    path: List[Tuple[int, int]]
    # Hazard type ("puddle", "obstacle", "both", "none")
    hazard_type: str
    # Detailed warning message
    warning_message: str
    # Puddle depth assessment ("safe", "warning", "critical")
    puddle_severity: str = "safe"


class PathPlanner:
    """
    Plans safe walking paths using A* algorithm.
    Analyzes hazard map and finds optimal route avoiding puddles and obstacles.
    """
    
    def __init__(self, config: PathPlanningConfig):
        """
        Initialize path planner.
        
        Args:
            config: Path planning configuration
        """
        self.logger = get_logger("path_planner")
        self.config = config
        self._app_config = get_config()
        
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols
        
        # Precompute turn angle thresholds
        self._angle_thresholds = sorted([
            (abs(v), k, v) 
            for k, v in config.turn_angles.items()
        ], reverse=True)
        
        # State for hysteresis
        self._last_direction = "straight"
        self._last_angle = 0.0
        
        self.logger.info("PathPlanner initialized")
    
    def _get_puddle_thresholds(self) -> Tuple[float, float]:
        """Get current puddle thresholds based on shoe analysis."""
        return (
            self._app_config.puddle_warning_threshold,
            self._app_config.puddle_critical_threshold
        )
    
    def plan(
        self,
        water_mask: np.ndarray,
        obstacles: List[BoundingBox],
        frame_shape: Tuple[int, ...]
    ) -> NavigationResult:
        """
        Plan safe navigation path.
        
        Args:
            water_mask: Binary mask of water regions
            obstacles: List of detected obstacles
            frame_shape: Shape of camera frame (H, W, C)
            
        Returns:
            NavigationResult with recommended action
        """
        frame_h, frame_w = frame_shape[:2]
        
        # Build hazard grid
        hazard_grid = self._build_hazard_grid(water_mask, obstacles, frame_h, frame_w)
        
        # Calculate weights for A*
        weight_grid = self._calculate_weights(hazard_grid)
        
        # Find path using A*
        start = (self.grid_rows - 1, self.grid_cols // 2)  # Bottom center
        goal = (0, self.grid_cols // 2)  # Top center
        
        path = self._astar(weight_grid, start, goal)
        
        # Extract turn recommendation from path
        if path and len(path) >= 2:
            turn_angle = self._calculate_turn_angle(path)
            direction = self._angle_to_direction(turn_angle)
        else:
            # No path found - try to find safest direction
            turn_angle, direction = self._find_safest_direction(hazard_grid)
        
        # Apply hysteresis
        turn_angle, direction = self._apply_hysteresis(turn_angle, direction)
        
        # Calculate safety score
        safety_score = self._calculate_safety(hazard_grid, path)
        
        # Determine hazard info
        hazard_type, hazard_dist = self._analyze_hazards(hazard_grid, water_mask, obstacles)
        
        # Assess puddle severity based on shoe thresholds
        puddle_severity = self._assess_puddle_severity(water_mask, frame_h)
        
        # Generate warning message
        should_stop = safety_score < self.config.safety_critical
        warning = self._generate_warning(safety_score, hazard_type, direction, puddle_severity)
        
        return NavigationResult(
            turn_angle=turn_angle,
            direction=direction,
            safety_score=safety_score,
            hazard_distance=hazard_dist,
            should_stop=should_stop,
            path=path,
            hazard_type=hazard_type,
            warning_message=warning,
            puddle_severity=puddle_severity
        )
    
    def _build_hazard_grid(
        self,
        water_mask: np.ndarray,
        obstacles: List[BoundingBox],
        frame_h: int,
        frame_w: int
    ) -> np.ndarray:
        """Build hazard grid from detections."""
        # Focus on ground level (bottom portion of frame)
        focus_start = int(frame_h * (1 - self.config.frame_focus_ratio))
        
        focus_h = frame_h - focus_start
        cell_h = focus_h // self.grid_rows
        cell_w = frame_w // self.grid_cols
        
        hazard_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        for row in range(self.grid_rows):
            y_start = focus_start + row * cell_h
            y_end = y_start + cell_h
            
            for col in range(self.grid_cols):
                x_start = col * cell_w
                x_end = x_start + cell_w
                
                # Water coverage
                if water_mask.size > 0:
                    water_cell = water_mask[y_start:y_end, x_start:x_end]
                    puddle_ratio = np.mean(water_cell) if water_cell.size > 0 else 0
                else:
                    puddle_ratio = 0
                
                # Obstacle coverage
                obstacle_ratio = 0
                for obs in obstacles:
                    # Check if obstacle overlaps this cell
                    ox, oy, ow, oh = obs.bbox
                    
                    # Calculate intersection
                    ix_start = max(x_start, ox)
                    iy_start = max(y_start, oy)
                    ix_end = min(x_end, ox + ow)
                    iy_end = min(y_end, oy + oh)
                    
                    if ix_end > ix_start and iy_end > iy_start:
                        intersection = (ix_end - ix_start) * (iy_end - iy_start)
                        cell_area = cell_h * cell_w
                        obstacle_ratio = max(obstacle_ratio, intersection / cell_area)
                
                # Combined hazard
                hazard = max(puddle_ratio, obstacle_ratio * 1.5)  # Obstacles weighted higher
                hazard_grid[row, col] = min(hazard, 1.0)
        
        return hazard_grid
    
    def _calculate_weights(self, hazard_grid: np.ndarray) -> np.ndarray:
        """Calculate A* weights from hazard grid."""
        weights = np.full_like(hazard_grid, self.config.base_weight)
        
        # Add hazard penalties
        puddle_mask = hazard_grid > self.config.puddle_coverage_threshold
        weights[puddle_mask] += self.config.puddle_penalty * hazard_grid[puddle_mask]
        
        # Edge penalty (prefer center)
        for col in range(self.grid_cols):
            edge_factor = abs(col - self.grid_cols // 2) / (self.grid_cols // 2)
            weights[:, col] += self.config.edge_penalty * edge_factor
        
        # Forward bonus (encourage moving forward)
        for row in range(self.grid_rows):
            forward_factor = 1.0 - (row / self.grid_rows)
            weights[row, :] += self.config.forward_bonus * forward_factor
        
        return weights
    
    def _astar(
        self,
        weights: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
        rows, cols = weights.shape
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        # Track visited
        closed_set = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            # Explore neighbors (8-directional)
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate cost (diagonal moves cost more)
                move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = g_score[current] + weights[neighbor] * move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        
        # No path found - return empty
        return []
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(
        self,
        came_from: dict,
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _calculate_turn_angle(self, path: List[Tuple[int, int]]) -> float:
        """Calculate recommended turn angle from path."""
        if len(path) < 2:
            return 0.0
        
        # Look at first few steps of path
        look_ahead = min(3, len(path) - 1)
        
        # Calculate horizontal deviation
        start_col = path[0][1]
        target_col = path[look_ahead][1]
        
        # Convert to angle
        col_diff = target_col - start_col
        max_diff = self.grid_cols // 2
        
        # Map to angle range
        angle = (col_diff / max_diff) * 30.0  # Max 30 degrees
        
        return angle
    
    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to direction name."""
        abs_angle = abs(angle)
        
        if abs_angle < 7.5:
            return "straight"
        elif abs_angle < 22.5:
            return "slight_left" if angle < 0 else "slight_right"
        else:
            return "hard_left" if angle < 0 else "hard_right"
    
    def _apply_hysteresis(
        self,
        angle: float,
        direction: str
    ) -> Tuple[float, str]:
        """Apply hysteresis to prevent oscillation."""
        angle_diff = abs(angle - self._last_angle)
        
        if angle_diff < self.config.direction_hysteresis:
            # Keep previous direction if change is small
            return self._last_angle, self._last_direction
        
        # Update state
        self._last_angle = angle
        self._last_direction = direction
        
        return angle, direction
    
    def _find_safest_direction(
        self,
        hazard_grid: np.ndarray
    ) -> Tuple[float, str]:
        """Find safest direction when no path exists."""
        bottom_row = hazard_grid[-1, :]
        
        # Find column with minimum hazard
        safest_col = np.argmin(bottom_row)
        center_col = self.grid_cols // 2
        
        col_diff = safest_col - center_col
        angle = (col_diff / (self.grid_cols // 2)) * 30.0
        
        direction = self._angle_to_direction(angle)
        
        return angle, direction
    
    def _calculate_safety(
        self,
        hazard_grid: np.ndarray,
        path: List[Tuple[int, int]]
    ) -> float:
        """Calculate safety score for planned path."""
        if not path:
            # No path = unsafe
            return 0.3
        
        # Calculate path hazard
        path_hazards = [hazard_grid[r, c] for r, c in path]
        mean_hazard = np.mean(path_hazards)
        max_hazard = np.max(path_hazards)
        
        # Safety is inverse of hazard
        safety = 1.0 - (0.6 * mean_hazard + 0.4 * max_hazard)
        
        return max(0.0, min(1.0, safety))
    
    def _analyze_hazards(
        self,
        hazard_grid: np.ndarray,
        water_mask: np.ndarray,
        obstacles: List[BoundingBox]
    ) -> Tuple[str, float]:
        """Analyze hazards to determine type and distance."""
        has_water = np.any(water_mask > 0)
        has_obstacles = len(obstacles) > 0
        
        if has_water and has_obstacles:
            hazard_type = "both"
        elif has_water:
            hazard_type = "puddle"
        elif has_obstacles:
            hazard_type = "obstacle"
        else:
            hazard_type = "none"
        
        # Find distance to nearest hazard (bottom row = close)
        if hazard_type != "none":
            for row in range(self.grid_rows - 1, -1, -1):
                if np.max(hazard_grid[row, :]) > 0.3:
                    distance = row / self.grid_rows
                    return hazard_type, distance
        
        return hazard_type, 1.0
    
    def _generate_warning(
        self,
        safety: float,
        hazard_type: str,
        direction: str,
        puddle_severity: str = "safe"
    ) -> str:
        """Generate appropriate warning message based on hazards and shoe thresholds."""
        if safety < self.config.safety_critical:
            return "stop"
        elif safety < self.config.safety_caution:
            if hazard_type == "puddle":
                # Use shoe-aware puddle warnings
                if puddle_severity == "critical":
                    return "puddle_deep"
                elif puddle_severity == "warning":
                    return "puddle_ahead"
                else:
                    return "puddle_shallow"
            elif hazard_type == "obstacle":
                return "obstacle_ahead"
            else:
                return "caution"
        else:
            if direction == "straight":
                return "clear"
            elif "left" in direction:
                return "turn_left" if "hard" in direction else "turn_slight_left"
            else:
                return "turn_right" if "hard" in direction else "turn_slight_right"
    
    def _assess_puddle_severity(
        self,
        water_mask: np.ndarray,
        frame_h: int
    ) -> str:
        """
        Assess puddle severity based on shoe thresholds.
        
        Args:
            water_mask: Binary mask of water regions
            frame_h: Frame height
            
        Returns:
            Severity level: "safe", "warning", or "critical"
        """
        if water_mask is None or not np.any(water_mask > 0):
            return "safe"
        
        # Get shoe-based thresholds
        warning_threshold, critical_threshold = self._get_puddle_thresholds()
        
        # Calculate puddle coverage in the bottom portion of frame (near user)
        focus_start = int(frame_h * (1 - self.config.frame_focus_ratio))
        bottom_region = water_mask[focus_start:, :]
        
        if bottom_region.size == 0:
            return "safe"
        
        # Calculate coverage ratio
        coverage = np.sum(bottom_region > 0) / bottom_region.size
        
        # Estimate puddle "depth" based on coverage and position
        # Larger, closer puddles are assumed deeper
        # Find the lowest (closest) row with water
        water_rows = np.where(np.any(water_mask > 0, axis=1))[0]
        if len(water_rows) == 0:
            return "safe"
        
        closest_water_row = water_rows[-1]  # Bottom-most row with water
        relative_distance = closest_water_row / frame_h
        
        # Combine coverage and distance for severity estimate
        # Closer puddles with higher coverage = more severe
        severity_score = coverage * (0.5 + 0.5 * relative_distance)
        
        # Compare against shoe-based thresholds
        if severity_score > critical_threshold * 2:
            return "critical"
        elif severity_score > warning_threshold:
            return "warning"
        else:
            return "safe"
