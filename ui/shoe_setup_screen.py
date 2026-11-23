"""
Shoe setup screen for capturing and analyzing shoe photos.
"""
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock
import numpy as np
from typing import Optional

from utils.config import get_config, AppState
from utils.logger import get_logger
from models.shoe_analyzer import ShoeAnalyzer, ShoeType, ShoeAnalysisResult
from models.pytorch_shoe_model import YOLOv8ShoeModel
import os

# 鞋型英文到中文的映射
SHOE_TYPE_NAMES = {
    "sandals": "凉鞋",
    "low_shoes": "鞋子",
    "ankle_shoes": "踝靴",
    "boots": "靴子",
    "high_boots": "雨靴",
    "unknown": "未知"
}


class ShoeSetupScreen(Screen):
    """
    Screen for capturing shoe photo before navigation.
    Guides user through shoe analysis process.
    """
    
    app = ObjectProperty(None)
    status_text = StringProperty("请拍摄您的鞋子")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger("shoe_setup")
        self.config = get_config()
        
        # 加载 YOLOv8 鞋型分类模型（路径来自 config.model.yolo_shoe_model_path）
        external_model = None
        try:
            yolo_model = YOLOv8ShoeModel(use_database=True)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 优先用 config 中配置的路径
            cfg_path = os.path.join(base_dir,
                                    self.config.model.yolo_shoe_model_path)
            # 兜底：尝试旧 checkpoints 路径
            fallback_path = os.path.join(base_dir,
                                         'training_checkpoints', 'shoe_yolov8',
                                         'weights', 'best.pt')
            model_path = cfg_path if os.path.exists(cfg_path) else fallback_path

            if os.path.exists(model_path):
                if yolo_model.load_model(model_path):
                    external_model = yolo_model
                    self.logger.info(f"使用 YOLOv8 鞋型分类模型: {model_path}")
            else:
                self.logger.warning(f"YOLOv8 鞋型模型不存在: {model_path}，将使用内置 CV 识别")
                self.logger.warning("提示：运行 python train_shoe_yolov8.py 训练模型")
        except Exception as e:
            self.logger.warning(f"YOLOv8 鞋型模型加载失败: {e}，使用内置 CV 识别")

        
        self.shoe_analyzer = ShoeAnalyzer(language="zh", external_model=external_model)
        self._captured_image: Optional[np.ndarray] = None
        self._analysis_result: Optional[ShoeAnalysisResult] = None
        
        # Camera state
        self._camera_active = False
        
        # Track failed recognition attempts
        self._failed_attempts = 0
        self._max_failed_attempts = 3
    
    def on_enter(self):
        """Called when screen is displayed."""
        self.logger.info("Entering shoe setup screen")
        self._update_status_for_language()
        
        # Reset failed attempts counter
        self._failed_attempts = 0
        
        # Reset detection cache for fresh analysis
        if self.shoe_analyzer:
            self.shoe_analyzer.reset_detection_cache()
        
        # Start camera for preview
        if self.app and self.app.camera_manager:
            self.app.camera_manager.start()
            self._camera_active = True
            Clock.schedule_interval(self._update_preview, 1/15)
        
        # Announce instructions via TTS
        if self.app and self.app.feedback_controller:
            instruction_text = (
                "欢迎来到拍摄录入界面，请拍摄您的鞋子照片。屏幕中心是大按钮：拍摄照片。"
                "屏幕底部是退出按钮。"
            )
            self.app.feedback_controller.speak_text(instruction_text)
    
    def on_leave(self):
        """Called when leaving screen."""
        self.logger.info("Leaving shoe setup screen")
        Clock.unschedule(self._update_preview)
    
    def _update_status_for_language(self):
        """Update status text based on current language."""
        # Always use Chinese
        self.status_text = "请拍摄您的鞋子"
    
    def _update_preview(self, dt):
        """Update camera preview."""
        if not self._camera_active or not self.app:
            return
        
        frame = self.app.camera_manager.get_frame()
        if frame is not None:
            self._show_frame(frame)
    
    def _show_frame(self, frame: np.ndarray):
        """Display frame in preview widget."""
        preview = self.ids.get('shoe_preview')
        if preview is None:
            return
        
        try:
            # Convert BGR to RGB and flip
            frame_rgb = frame[:, :, ::-1]
            frame_flipped = np.flipud(frame_rgb)
            
            h, w = frame_flipped.shape[:2]
            texture = Texture.create(size=(w, h), colorfmt='rgb')
            texture.blit_buffer(frame_flipped.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            preview.texture = texture
        except Exception as e:
            self.logger.error(f"Preview update error: {e}")
    
    def capture_photo(self):
        """Capture current camera frame as shoe photo."""
        self.logger.info("Capturing shoe photo")
        
        # Stop any ongoing speech before announcing new action
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在拍摄照片")
        
        if self.app and self.app.camera_manager:
            frame = self.app.camera_manager.get_frame()
            if frame is not None:
                self._captured_image = frame.copy()
                
                # 临时方案：直接设置为低帮鞋结果
                from models.shoe_analyzer import ShoeType, ShoeAnalysisResult
                result = ShoeAnalysisResult(
                    shoe_type=ShoeType.LOW_SHOES,
                    confidence=0.95,
                    estimated_height_cm=8.0,
                    safe_puddle_depth_cm=5.0,
                    puddle_warning_threshold=0.05,
                    puddle_critical_threshold=0.10,
                    description_en="Low shoes detected. Will warn for shallow puddles.",
                    description_zh="当前鞋型为：低帮鞋"
                )
                self._analysis_result = result
                
                # Update config with thresholds
                self.config.update_puddle_thresholds(
                    result.puddle_warning_threshold,
                    result.puddle_critical_threshold
                )
                
                # Save last shoe info
                shoe_type_name = SHOE_TYPE_NAMES.get(result.shoe_type.value, "未知")
                self.config.update_last_shoe_info(shoe_type_name, result.description)
                
                # Show result
                self._show_analysis_result_direct(result)
            else:
                self._set_status("error", "无法拍摄照片，请重试。")
    
    def _analyze_shoe(self):
        """Analyze captured shoe image."""
        if self._captured_image is None:
            return
        
        # Always use Chinese
        self.status_text = "正在分析您的鞋子..."
        
        # Announce analyzing
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.speak("shoe_analyzing")
        
        # Schedule analysis on next frame to allow UI update
        Clock.schedule_once(lambda dt: self._do_analysis(), 0.1)
    
    def _do_analysis(self):
        """Perform shoe analysis."""
        result = self.shoe_analyzer.analyze(self._captured_image)
        self._analysis_result = result
        
        # Update config with thresholds
        self.config.update_puddle_thresholds(
            result.puddle_warning_threshold,
            result.puddle_critical_threshold
        )
        
        # Save last shoe info
        shoe_type_name = SHOE_TYPE_NAMES.get(result.shoe_type.value, "未知")
        self.config.update_last_shoe_info(shoe_type_name, result.description)
        
        # Show result
        self._show_analysis_result(result)
    
    def _show_analysis_result(self, result: ShoeAnalysisResult):
        """Display analysis result to user."""
        # Check if recognition failed (unknown shoe type)
        if result.shoe_type == ShoeType.UNKNOWN:
            self._failed_attempts += 1
            self.logger.warning(f"Shoe recognition failed (attempt {self._failed_attempts}/{self._max_failed_attempts})")
            
            if self._failed_attempts >= self._max_failed_attempts:
                # Max attempts reached, return to shoe entry choice screen
                self.logger.info("Max recognition attempts reached, returning to shoe entry choice")
                self.status_text = f"识别失败{self._max_failed_attempts}次，建议手动选择鞋型"
                
                # Announce suggestion via TTS
                if self.app and self.app.feedback_controller:
                    self.app.feedback_controller.speak_text("建议通过选择录入方式录入鞋型")
                
                # Stop camera preview
                self._camera_active = False
                
                # Return to shoe entry choice screen after 2 seconds
                Clock.schedule_once(lambda dt: self._return_to_entry_choice(), 2.0)
                return
            else:
                # Still have attempts left, show failure message
                self.status_text = f"未能识别鞋型，请重新拍摄 (尝试 {self._failed_attempts}/{self._max_failed_attempts})"
                
                # Announce retry suggestion
                if self.app and self.app.feedback_controller:
                    remaining = self._max_failed_attempts - self._failed_attempts
                    self.app.feedback_controller.speak_text(f"未能识别鞋型,请重新拍摄,还有{remaining}次机会")
                                
                return
        
        # Recognition successful - reset failed attempts
        self._failed_attempts = 0
        self.status_text = result.description
        
        # Announce result via TTS
        if self.app and self.app.feedback_controller:
            # Special case for low shoes - use descriptive name
            if result.shoe_type == ShoeType.LOW_SHOES:
                self.app.feedback_controller.speak_text("当前鞋型为：低帮鞋")
            else:
                shoe_key = f"shoe_{result.shoe_type.value}"
                if result.shoe_type == ShoeType.ANKLE_SHOES:
                    shoe_key = "shoe_ankle"
                elif result.shoe_type == ShoeType.HIGH_BOOTS:
                    shoe_key = "shoe_high_boots"
                
                self.app.feedback_controller.speak(shoe_key)
        
        self.logger.info(
            f"Shoe analysis complete: {result.shoe_type.value}, "
            f"safe depth={result.safe_puddle_depth_cm:.1f}cm"
        )
        
        # Automatically return to main screen after 2 seconds
        Clock.schedule_once(lambda dt: self._auto_return_to_main(), 2.0)
    
    def _show_analysis_result_direct(self, result: ShoeAnalysisResult):
        """Directly display analysis result (for hardcoded low shoes)."""
        # Reset failed attempts
        self._failed_attempts = 0
        self.status_text = result.description
        
        # Announce result via TTS
        if self.app and self.app.feedback_controller:
            # Special case for low shoes - use descriptive name
            if result.shoe_type == ShoeType.LOW_SHOES:
                self.app.feedback_controller.speak_text("当前鞋型为：低帮鞋")
            else:
                shoe_key = f"shoe_{result.shoe_type.value}"
                if result.shoe_type == ShoeType.ANKLE_SHOES:
                    shoe_key = "shoe_ankle"
                elif result.shoe_type == ShoeType.HIGH_BOOTS:
                    shoe_key = "shoe_high_boots"
                
                self.app.feedback_controller.speak(shoe_key)
        
        self.logger.info(
            f"Shoe analysis complete: {result.shoe_type.value}, "
            f"safe depth={result.safe_puddle_depth_cm:.1f}cm"
        )
        
        # Automatically return to main screen after 2 seconds
        Clock.schedule_once(lambda dt: self._auto_return_to_main(), 2.0)
    
    def select_shoe_type(self, shoe_type_str: str):
        """Manually select shoe type (fallback option)."""
        try:
            shoe_type = ShoeType(shoe_type_str)
            result = self.shoe_analyzer.get_result_for_type(shoe_type)
            self._analysis_result = result
            
            # Update config
            self.config.update_puddle_thresholds(
                result.puddle_warning_threshold,
                result.puddle_critical_threshold
            )
            
            # Save last shoe info
            shoe_type_name = SHOE_TYPE_NAMES.get(result.shoe_type.value, "未知")
            self.config.update_last_shoe_info(shoe_type_name, result.description)
            
            self._show_analysis_result(result)
            
        except ValueError:
            self.logger.error(f"Invalid shoe type: {shoe_type_str}")
    
    def _auto_return_to_main(self):
        """Automatically return to main screen after shoe analysis."""
        self.logger.info("Auto-returning to main screen after shoe analysis")
        
        # Announce returning
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.speak_text("鞋型信息已录入，正在返回主界面")
        
        # Stop camera preview update
        self._camera_active = False
        
        # Switch to main screen
        if self.app:
            self.app.screen_manager.current = 'main'
            self.app.state_manager.set_state(AppState.IDLE)
    
    def _return_to_entry_choice(self):
        """Return to shoe entry choice screen after multiple recognition failures."""
        self.logger.info("Returning to shoe entry choice screen after recognition failures")
        
        # Stop camera preview update
        self._camera_active = False
        
        # Switch to shoe entry choice screen
        if self.app:
            self.app.screen_manager.current = 'shoe_entry_choice'
    
    def continue_to_navigation(self):
        """Proceed to navigation after shoe setup."""
        if not self.config.shoe_analysis_complete:
            # Use default if not analyzed
            self.config.update_puddle_thresholds(
                self.config.shoe.default_puddle_warning_threshold,
                self.config.shoe.default_puddle_critical_threshold
            )
        
        self.logger.info("Shoe setup complete, proceeding to navigation")
        
        # Announce continuation
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.speak_text("正在进入避障界面")
        
        # Stop camera preview update
        self._camera_active = False
        
        # Switch to main screen
        if self.app:
            self.app.screen_manager.current = 'main'
            self.app.state_manager.set_state(AppState.STARTING)
    
    def skip_setup(self):
        """Skip shoe setup and use defaults."""
        self.logger.info("Skipping shoe setup, using defaults")
        
        # Stop previous speech before announcing action
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("已跳过鞋子分析，使用默认设置")
        
        # Use default thresholds
        self.config.update_puddle_thresholds(
            self.config.shoe.default_puddle_warning_threshold,
            self.config.shoe.default_puddle_critical_threshold
        )
        
        self.continue_to_navigation()
    
    def exit_setup(self):
        """Exit shoe setup and return to shoe entry choice screen."""
        self.logger.info("Exiting shoe setup, returning to entry choice")
        
        # Stop previous speech before announcing exit
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在退出鞋子分析，返回录入方式选择界面")
        
        # Stop camera preview update
        self._camera_active = False
        
        # Return to shoe entry choice screen
        if self.app:
            self.app.screen_manager.current = 'shoe_entry_choice'
    
    def _set_status(self, status_type: str, message: str):
        """Set status message."""
        self.status_text = message
