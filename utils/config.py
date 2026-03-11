"""
Configuration constants for Blind Navigation Assistant.
Centralized settings for all modules.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class AppState(Enum):
    """Application state machine states."""
    IDLE = "idle"
    SHOE_SETUP = "shoe_setup"  # Waiting for shoe photo
    STARTING = "starting"
    ACTIVE = "active"
    PROCESSING = "processing"
    WARNING = "warning"
    PAUSED = "paused"
    ERROR = "error"


class HazardLevel(Enum):
    """Hazard severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CameraConfig:
    """Camera capture configuration."""
    # Resolution (width, height)
    resolution: Tuple[int, int] = (640, 480)
    # Target frames per second
    target_fps: int = 15
    # Maximum frames in queue before dropping
    max_queue_size: int = 3
    # Camera index (0 = default camera)
    camera_index: int = 0
    # Horizontal field of view in degrees
    fov_horizontal: float = 60.0


@dataclass
class ModelConfig:
    """ML model configuration."""
    # DeepLabV3 settings
    deeplabv3_input_size: Tuple[int, int] = (256, 256)
    deeplabv3_model_path: str = "assets/models/deeplabv3_water.tflite"
    water_class_index: int = 0  # Index of water class in segmentation output
    water_confidence_threshold: float = 0.15
    
    # YOLOv8 settings
    yolo_input_size: Tuple[int, int] = (256, 256)
    yolo_model_path: str = "assets/models/yolov8_obstacle.tflite"
    yolo_confidence_threshold: float = 0.4
    yolo_nms_threshold: float = 0.45
    
    # Classes to detect as obstacles (对应训练数据集 dataset.yaml 的 11 个类别)
    obstacle_classes: List[int] = field(default_factory=lambda: [
        0,   # person 行人
        1,   # bicycle 自行车
        2,   # car 汽车
        3,   # motorcycle 摩托车
        4,   # stairs 台阶
        5,   # wire 电线
        6,   # bench 长椅
        7,   # speed_bump 减速带
        8,   # pothole 坑洞
        9,   # utility_pole 电线杆
        10,  # tree 树木
    ])
    
    # Use GPU acceleration if available
    use_gpu_delegate: bool = True
    # Number of CPU threads for inference
    num_threads: int = 2


@dataclass
class ReflectionConfig:
    """Traditional CV reflection detection configuration."""
    # HSV thresholds for water reflection
    hsv_saturation_min: int = 20
    hsv_saturation_max: int = 255
    hsv_value_min: int = 150
    hsv_value_max: int = 255
    
    # Edge detection thresholds
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    
    # Minimum contour area to consider as puddle (pixels)
    min_contour_area: int = 500
    
    # Morphological operation kernel size
    morph_kernel_size: int = 5


@dataclass
class ShoeConfig:
    """Shoe analysis configuration."""
    # Whether shoe analysis is required before navigation
    require_shoe_analysis: bool = False
    
    # Default puddle warning threshold (if no shoe analyzed)
    default_puddle_warning_threshold: float = 0.04  # ~2cm
    default_puddle_critical_threshold: float = 0.02  # ~1cm
    
    # Puddle depth estimation parameters
    # cm of water depth per fraction of frame height
    cm_per_frame_ratio: float = 50.0
    
    # Allow manual shoe type selection as fallback
    allow_manual_selection: bool = True


@dataclass
class PathPlanningConfig:
    """Path planning algorithm configuration."""
    # Grid dimensions
    grid_rows: int = 8
    grid_cols: int = 8
    
    # Focus on bottom portion of frame (ground level)
    frame_focus_ratio: float = 0.67  # Bottom 2/3 of frame
    
    # Weight penalties
    base_weight: float = 1.0
    puddle_penalty: float = 10.0
    obstacle_penalty: float = 20.0
    edge_penalty: float = 2.0
    forward_bonus: float = -0.5  # Negative = encourages forward
    
    # Puddle coverage threshold (cell covered > this % = penalty)
    puddle_coverage_threshold: float = 0.3
    
    # Safety thresholds
    safety_critical: float = 0.4   # Below this = stop
    safety_caution: float = 0.7    # Below this = caution
    
    # Turn angle quantization (degrees)
    turn_angles: Dict[str, float] = field(default_factory=lambda: {
        "hard_left": -30.0,
        "slight_left": -15.0,
        "straight": 0.0,
        "slight_right": 15.0,
        "hard_right": 30.0,
    })
    
    # Hysteresis to prevent oscillation (minimum change to update direction)
    direction_hysteresis: float = 10.0


@dataclass
class AudioConfig:
    """Audio guidance configuration."""
    # TTS settings
    speech_rate: float = 1.3  # Speed multiplier
    speech_volume: float = 0.8
    
    # Minimum interval between repeated messages (seconds)
    min_message_interval: float = 2.0
    
    # Priority levels (higher = more urgent)
    priority_normal: int = 1
    priority_warning: int = 2
    priority_critical: int = 3
    
    # Audio phrases (English)
    phrases: Dict[str, str] = field(default_factory=lambda: {
        "starting": "Navigation assistant starting",
        "ready": "Ready. Point camera forward to begin",
        "clear": "Clear ahead",
        "puddle_ahead": "Puddle ahead",
        "puddle_left": "Puddle on left",
        "puddle_right": "Puddle on right",
        "puddle_shallow": "Shallow puddle ahead, safe to cross",
        "puddle_deep": "Deep puddle ahead, please avoid",
        "obstacle_ahead": "Obstacle ahead",
        "obstacle_left": "Obstacle on left",
        "obstacle_right": "Obstacle on right",
        "turn_left": "Turn left",
        "turn_slight_left": "Slight left",
        "turn_right": "Turn right",
        "turn_slight_right": "Slight right",
        "go_straight": "Go straight",
        "caution": "Caution, proceed slowly",
        "stop": "Stop! Hazards ahead",
        "paused": "Navigation paused",
        "resumed": "Navigation resumed",
        "error": "Error occurred. Please restart",
        "camera_blocked": "Camera may be blocked",
        "low_battery": "Low battery, switching to power save mode",
        "shoe_prompt": "Please take a photo of your shoes for today",
        "shoe_analyzing": "Analyzing your shoes",
        "shoe_detected": "Shoes detected. Puddle warnings adjusted accordingly",
        "shoe_sandals": "Sandals detected. Will warn for very shallow puddles",
        "shoe_low": "Low shoes detected. Will warn for shallow puddles",
        "shoe_ankle": "Ankle shoes detected. Moderate puddle protection",
        "shoe_boots": "Boots detected. Good puddle protection",
        "shoe_high_boots": "Rain boots detected. High puddle protection",
        "shoe_unknown": "Could not identify shoes. Using default settings",
        "wake_listening": "Listening for wake word. Say bye bye puddle to start",
        "wake_detected": "Wake word detected. Starting navigation",
        "voice_command_stop": "Voice command received. Stopping navigation",
        "voice_command_start": "Voice command received. Starting navigation",
        "arrived": "Destination reached. Navigation assistant exiting",
        "navigation_completed": "Current navigation completed",
        "app_introduction": "Welcome to Blind Navigation Assistant. This software helps you avoid water puddles and obstacles in rainy weather. You can start by registering your shoes to calibrate detection. Say 'Xia yu le, wo yao chu men' to wake the assistant.",
    })
    
    # Chinese phrases (optional)
    phrases_zh: Dict[str, str] = field(default_factory=lambda: {
        "starting": "避障助手启动中",
        "ready": "准备就绪。请将摄像头向前",
        "clear": "前方畅通",
        "puddle_ahead": "前方有水坑",
        "puddle_left": "左侧有水坑",
        "puddle_right": "右侧有水坑",
        "puddle_shallow": "前方浅水坑，可以通过",
        "puddle_deep": "前方深水坑，请绕行",
        "obstacle_ahead": "前方有障碍物",
        "obstacle_left": "左侧有障碍物",
        "obstacle_right": "右侧有障碍物",
        "turn_left": "请向左转",
        "turn_slight_left": "请稍微向左",
        "turn_right": "请向右转",
        "turn_slight_right": "请稍微向右",
        "go_straight": "请直行",
        "caution": "注意，请慢行",
        "stop": "停止！前方有危险",
        "paused": "避障已暂停",
        "resumed": "避障已恢复",
        "error": "发生错误，请重启应用",
        "camera_blocked": "摄像头可能被遮挡",
        "low_battery": "电量低，切换到省电模式",
        "shoe_prompt": "请拍摄今天出门穿的鞋子照片",
        "shoe_analyzing": "正在分析您的鞋子",
        "shoe_detected": "已识别鞋子，已相应调整水坑警告",
        "shoe_sandals": "检测到凉鞋，将对非常浅的水坑发出警告",
        "shoe_low": "检测到低帮鞋，将对浅水坑发出警告",
        "shoe_ankle": "检测到踝靴，中等防水保护",
        "shoe_boots": "检测到靴子，良好的防水保护",
        "shoe_high_boots": "检测到雨靴，高度防水保护",
        "shoe_unknown": "无法识别鞋子，使用默认设置",
        "wake_listening": "正在监听唤醒词。说拜拜雨坑开始避障",
        "wake_detected": "检测到唤醒词。正在启动避障",
        "voice_command_stop": "收到语音指令。停止避障",
        "voice_command_start": "收到语音指令。开始避障",
        "arrived": "已到达目的地。正在退出避障系统",
        "navigation_completed": "本次避障已完成",
        "app_introduction": "欢迎使用不积洼步避障助手。本软件旨在帮助您在雨天出行时避开水坑与障碍物。您可以先录入您的鞋型，以便系统为您量身定制避障方案。您也可以直接对手机说：下雨了，我要出门，来唤醒导航助手。祝您出行平安。",
    })


@dataclass
class VoiceConfig:
    """Voice activation configuration."""
    # Enable voice activation
    enable_voice_activation: bool = True
    # Enable voice commands during navigation
    enable_voice_commands: bool = True
    # Recognition language
    recognition_language: str = "zh-CN"
    # Wake words
    wake_words: List[str] = field(default_factory=lambda: [
        "下雨了，我要出门",
    ])


@dataclass
class DistanceConfig:
    """Distance estimation configuration."""
    # Assumed camera height from ground (meters)
    camera_height_m: float = 1.2
    # Average step length (meters) - Updated to 0.5m (2 steps = 1 meter)
    step_length_m: float = 0.5
    # Enable precise distance guidance
    enable_precise_guidance: bool = True
    # Minimum distance to give precise guidance (meters)
    min_guidance_distance: float = 0.5
    # Maximum reliable distance estimation (meters)
    max_distance_m: float = 15.0


@dataclass
class HapticConfig:
    """Haptic feedback configuration."""
    # Vibration patterns (duration in milliseconds)
    # Format: List of (vibrate_ms, pause_ms) tuples
    patterns: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        "puddle_warning": [(150, 100), (150, 0)],      # Two short pulses
        "obstacle_warning": [(100, 50), (100, 50), (100, 0)],  # Three rapid pulses
        "critical_alert": [(500, 0)],                   # Long pulse
        "success": [(200, 0)],                          # Single gentle pulse
        "direction_change": [(100, 0)],                 # Quick tap
    })
    
    # Minimum interval between haptic feedback (seconds)
    min_haptic_interval: float = 0.5


@dataclass
class PerformanceConfig:
    """Performance and battery optimization settings."""
    # Adaptive frame rate settings
    fps_high_activity: int = 30
    fps_normal: int = 15
    fps_low_activity: int = 10
    fps_power_save: int = 8
    
    # Battery thresholds
    battery_warning_level: float = 0.2  # 20%
    battery_critical_level: float = 0.1  # 10%
    
    # Processing timeout (ms)
    inference_timeout: int = 200
    
    # Memory limits (MB)
    max_memory_usage: int = 400


@dataclass
class AppConfig:
    """Main application configuration combining all settings."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    haptic: HapticConfig = field(default_factory=HapticConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    shoe: ShoeConfig = field(default_factory=ShoeConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    
    # App-wide settings
    debug_mode: bool = False
    show_camera_preview: bool = True
    language: str = "zh"  # "en" or "zh"
    
    # Runtime shoe analysis result (set after analysis)
    shoe_analysis_complete: bool = False
    puddle_warning_threshold: float = 0.04  # Will be updated by shoe analysis
    puddle_critical_threshold: float = 0.02
    
    # Last recorded shoe info
    last_shoe_type: str = "无"  # 最近一次录入的鞋型
    last_shoe_description: str = ""  # 最近一次录入的鞋型描述
    
    # Track first time run
    is_first_run: bool = True
    
    def get_phrases(self) -> Dict[str, str]:
        """Get phrases in current language."""
        if self.language == "zh":
            return self.audio.phrases_zh
        return self.audio.phrases
    
    def update_puddle_thresholds(self, warning: float, critical: float) -> None:
        """Update puddle warning thresholds based on shoe analysis."""
        self.puddle_warning_threshold = warning
        self.puddle_critical_threshold = critical
        self.shoe_analysis_complete = True
    
    def update_last_shoe_info(self, shoe_type_name: str, description: str) -> None:
        """更新最近一次录入的鞋型信息。"""
        self.last_shoe_type = shoe_type_name
        self.last_shoe_description = description
        # 保存到文件
        self._save_shoe_info()
    
    def _get_shoe_info_path(self) -> str:
        """获取鞋型信息存储文件路径。"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        # 确保data目录存在
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'shoe_info.json')
    
    def _save_shoe_info(self) -> None:
        """保存鞋型信息到文件。"""
        try:
            data = {
                'last_shoe_type': self.last_shoe_type,
                'last_shoe_description': self.last_shoe_description,
                'puddle_warning_threshold': self.puddle_warning_threshold,
                'puddle_critical_threshold': self.puddle_critical_threshold,
                'is_first_run': self.is_first_run
            }
            with open(self._get_shoe_info_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving shoe info: {e}")
    
    def _load_shoe_info(self) -> None:
        """从文件加载鞋型信息。"""
        try:
            file_path = self._get_shoe_info_path()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.last_shoe_type = data.get('last_shoe_type', '无')
                    self.last_shoe_description = data.get('last_shoe_description', '')
                    self.puddle_warning_threshold = data.get('puddle_warning_threshold', 0.04)
                    self.puddle_critical_threshold = data.get('puddle_critical_threshold', 0.02)
                    self.is_first_run = data.get('is_first_run', True)
                    if self.last_shoe_type != '无':
                        self.shoe_analysis_complete = True
        except Exception as e:
            print(f"Error loading shoe info: {e}")


# Global configuration instance
config = AppConfig()
# Load saved shoe info on startup
config._load_shoe_info()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode."""
    config.debug_mode = enabled


def set_language(lang: str) -> None:
    """Set application language ('en' or 'zh')."""
    if lang in ("en", "zh"):
        config.language = lang


def get_asset_path(relative_path: str) -> str:
    """Get absolute path for an asset file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)
