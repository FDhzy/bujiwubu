"""
Precise guidance generator for detailed navigation instructions.
Generates step-by-step guidance like "Walk 3 steps forward, then 2 steps left".
"""
from typing import Optional, List
from dataclasses import dataclass

from utils.logger import get_logger
from utils.config import get_config, DistanceConfig
from guidance.distance_estimator import PuddleInfo, ObstacleInfo, NavigationGuidance


@dataclass
class PreciseInstruction:
    """Precise navigation instruction for TTS."""
    # Full instruction text (for TTS)
    text_en: str
    text_zh: str
    # Priority level
    priority: int  # 1=low, 2=normal, 3=high, 4=critical
    # Instruction type
    instruction_type: str  # "puddle", "obstacle", "clear", "stop"


class PreciseGuidanceGenerator:
    """
    Generates precise, step-by-step navigation instructions.
    Converts distance measurements to practical guidance.
    """
    
    def __init__(self, config: Optional[DistanceConfig] = None):
        """
        Initialize guidance generator.
        
        Args:
            config: Distance configuration
        """
        self.logger = get_logger("precise_guidance")
        self.config = config or get_config().distance
        self.step_length = self.config.step_length_m
        
        self.logger.info("PreciseGuidanceGenerator initialized")
    
    def generate_puddle_instruction(
        self,
        puddle: PuddleInfo,
        avoid_direction: str = "left"
    ) -> PreciseInstruction:
        """
        Generate precise instruction for puddle avoidance.
        
        Args:
            puddle: Puddle information with distance
            avoid_direction: Direction to avoid ("left" or "right")
            
        Returns:
            PreciseInstruction with bilingual text
        """
        distance = puddle.distance_m
        diameter = puddle.diameter_m
        
        # Calculate steps using the 2 steps per meter rule
        steps_forward = int(distance * 2)
        steps_side = max(1, int(diameter * 2))
        
        # Generate English text
        if distance < 1.0:
            text_en = (
                f"Warning! Puddle very close ahead. Move to the {avoid_direction} immediately."
            )
            priority = 4
        elif distance < 2.5:
            text_en = (
                f"Puddle ahead. Please walk forward, then move to the {avoid_direction} to avoid."
            )
            priority = 3
        else:
            text_en = (
                f"Puddle detected ahead. Please walk forward, then shift to the {avoid_direction}."
            )
            priority = 2
        
        # Generate Chinese text
        direction_zh = "左" if avoid_direction == "left" else "右"
        
        if distance < 1.0:
            text_zh = (
                f"注意！前方有水坑非常近。请立即向{direction_zh}绕行。"
            )
        elif distance < 2.5:
            text_zh = (
                f"前方有水坑，建议向前行走，再向{direction_zh}绕行避开。"
            )
        else:
            text_zh = (
                f"前方检测到水坑，建议向前行走，再向{direction_zh}绕行避开。"
            )
        
        return PreciseInstruction(
            text_en=text_en,
            text_zh=text_zh,
            priority=priority,
            instruction_type="puddle"
        )
    
    def generate_obstacle_instruction(
        self,
        obstacle: ObstacleInfo,
        avoid_direction: str = "left"
    ) -> PreciseInstruction:
        """
        Generate precise instruction for obstacle avoidance.
        
        Args:
            obstacle: Obstacle information with distance
            avoid_direction: Direction to avoid
            
        Returns:
            PreciseInstruction with bilingual text
        """
        distance = obstacle.distance_m
        width = obstacle.width_m
        obj_name = obstacle.class_name
        
        # Chinese object names
        obj_names_zh = {
            "person": "行人",
            "bicycle": "自行车",
            "car": "汽车",
            "motorcycle": "摩托车",
            "bus": "公交车",
            "truck": "卡车",
            "bench": "长椅",
            "chair": "椅子",
            "fire hydrant": "消防栓",
            "traffic light": "交通灯",
            "stop sign": "停车标志",
        }
        obj_name_zh = obj_names_zh.get(obj_name, "障碍物")
        
        # Calculate steps using the 2 steps per meter rule
        steps_forward = int(distance * 2)
        steps_side = max(1, int(width * 2))
        
        direction_zh = "左" if avoid_direction == "left" else "右"
        
        if distance < 1.5:
            text_en = (
                f"Warning! {obj_name.capitalize()} very close. "
                f"Move {steps_side} steps to the {avoid_direction} now."
            )
            text_zh = (
                f"警告！{obj_name_zh}非常近。"
                f"请立即向{direction_zh}移动{steps_side}步。"
            )
            priority = 4
        elif distance < 3.0:
            text_en = (
                f"{obj_name.capitalize()} ahead. Suggest walking {steps_forward} steps forward, then {steps_side} steps {avoid_direction}."
            )
            text_zh = (
                f"前方有{obj_name_zh}，建议向前走{steps_forward}步，再向{direction_zh}绕行{steps_side}步。"
            )
            priority = 3
        else:
            text_en = (
                f"{obj_name.capitalize()} detected ahead. Suggest walking {steps_forward} steps forward, then {steps_side} steps {avoid_direction}."
            )
            text_zh = (
                f"前方检测到{obj_name_zh}，建议向前走{steps_forward}步，再向{direction_zh}绕行{steps_side}步。"
            )
            priority = 2
        
        return PreciseInstruction(
            text_en=text_en,
            text_zh=text_zh,
            priority=priority,
            instruction_type="obstacle"
        )
    
    def generate_combined_instruction(
        self,
        puddles: List[PuddleInfo],
        obstacles: List[ObstacleInfo]
    ) -> PreciseInstruction:
        """
        Generate instruction considering both puddles and obstacles.
        
        Args:
            puddles: List of detected puddles
            obstacles: List of detected obstacles
            
        Returns:
            PreciseInstruction for the most critical hazard
        """
        # Find closest hazards
        closest_puddle = puddles[0] if puddles else None
        closest_obstacle = obstacles[0] if obstacles else None
        
        # Clear path
        if not closest_puddle and not closest_obstacle:
            return PreciseInstruction(
                text_en="Path is clear. Continue straight ahead.",
                text_zh="前方畅通，请继续直行。",
                priority=1,
                instruction_type="clear"
            )
        
        # Determine avoidance direction considering both hazards
        if closest_puddle and closest_obstacle:
            # Case 1: Hazards on opposite sides (Conflict)
            # If avoiding the obstacle would lead into the puddle, prioritize the obstacle per user requirement
            if (closest_puddle.direction == "left" and closest_obstacle.direction == "right") or \
               (closest_puddle.direction == "right" and closest_obstacle.direction == "left"):
                
                # If both are dangerously close, stop is still the safest choice
                if closest_puddle.distance_m < 1.2 and closest_obstacle.distance_m < 1.2:
                    return PreciseInstruction(
                        text_en="Stop! Hazards on both sides. Wait for guidance.",
                        text_zh="停止！两侧都有障碍。请等待指引。",
                        priority=4,
                        instruction_type="stop"
                    )
                
                # Conflict detected: Prioritize avoiding the obstacle
                avoid = "left" if closest_obstacle.direction == "right" else "right"
                return self.generate_obstacle_instruction(closest_obstacle, avoid)
            
            # Case 2: Hazards on the same side or both in center
            else:
                # Prioritize obstacle instruction if it is closer or within a similar range
                # Obstacles are generally higher physical risk than puddles
                if closest_obstacle.distance_m <= closest_puddle.distance_m + 1.0:
                    avoid = "right" if closest_obstacle.direction in ("left", "center") else "left"
                    return self.generate_obstacle_instruction(closest_obstacle, avoid)
                else:
                    avoid = "right" if closest_puddle.direction in ("left", "center") else "left"
                    return self.generate_puddle_instruction(closest_puddle, avoid)
        
        # Only puddle
        if closest_puddle:
            avoid = "right" if closest_puddle.direction in ("left", "center") else "left"
            return self.generate_puddle_instruction(closest_puddle, avoid)
        
        # Only obstacle
        avoid = "right" if closest_obstacle.direction in ("left", "center") else "left"
        return self.generate_obstacle_instruction(closest_obstacle, avoid)
    
    def get_text(self, instruction: PreciseInstruction, language: str = "zh") -> str:
        """Get instruction text in specified language."""
        return instruction.text_zh if language == "zh" else instruction.text_en
