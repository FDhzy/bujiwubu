"""
Shoe type choice screen - 鞋型选择界面
"""
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.clock import Clock
from utils.logger import get_logger
from utils.config import get_config, AppState
from models.shoe_analyzer import ShoeAnalyzer, ShoeType
import os

# Load KV file
kv_path = os.path.join(os.path.dirname(__file__), 'shoe_type_choice_screen.kv')
Builder.load_file(kv_path)

# 鞋型英文到中文的映射
SHOE_TYPE_NAMES = {
    "sandals": "凉鞋",
    "low_shoes": "低帮鞋",
    "ankle_shoes": "踝靴",
    "boots": "靴子",
    "high_boots": "雨靴",
    "unknown": "未知"
}


class ShoeTypeChoiceScreen(Screen):
    """
    Screen for choosing shoe type manually.
    User can select from sandals, shoes, boots, or rain boots.
    """
    
    app = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger("shoe_type_choice")
        self.shoe_analyzer = ShoeAnalyzer(language="zh")
        self._selected_result = None  # Store selected shoe analysis result
        self.config = get_config()
    
    def on_enter(self):
        """Called when screen is displayed."""
        self.logger.info("Entering shoe type choice screen")
        
        # Reset status
        status_label = self.ids.get('status_label')
        if status_label:
            status_label.text = "请选择您的鞋型"
        
        self._selected_result = None
        
        # Announce instructions via TTS
        if self.app and self.app.feedback_controller:
            instruction_text = (
                "欢迎来到鞋型选择界面，请选择您的鞋型。屏幕中心分布着四个大按钮："
                "左上方是凉鞋，右上方是低帮鞋，左下方是靴子，右下方是雨靴。"
                "屏幕底部是退出按钮。"
            )
            self.app.feedback_controller.speak_text(instruction_text)
    
    def on_shoe_type_selected(self, shoe_type_str: str):
        """User selected a shoe type."""
        self.logger.info(f"User selected shoe type: {shoe_type_str}")
        
        # Stop any ongoing speech before announcing new selection
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
        
        try:
            shoe_type = ShoeType(shoe_type_str)
            result = self.shoe_analyzer.get_result_for_type(shoe_type)
            self._selected_result = result
            
            # Update config with thresholds immediately
            if self.app:
                self.app.app_config.update_puddle_thresholds(
                    result.puddle_warning_threshold,
                    result.puddle_critical_threshold
                )
            
            # Save last shoe info immediately
            shoe_type_name = SHOE_TYPE_NAMES.get(result.shoe_type.value, "未知")
            self.config.update_last_shoe_info(shoe_type_name, result.description)
            
            # Update status label with description
            status_label = self.ids.get('status_label')
            if status_label:
                status_label.text = result.description
            
            # Announce result via TTS
            if self.app and self.app.feedback_controller:
                shoe_key = f"shoe_{result.shoe_type.value}"
                if result.shoe_type == ShoeType.UNKNOWN:
                    shoe_key = "shoe_unknown"
                elif result.shoe_type == ShoeType.LOW_SHOES:
                    shoe_key = "shoe_low"
                elif result.shoe_type == ShoeType.ANKLE_SHOES:
                    shoe_key = "shoe_ankle"
                elif result.shoe_type == ShoeType.HIGH_BOOTS:
                    shoe_key = "shoe_high_boots"
                
                self.app.feedback_controller.speak(shoe_key)
            
            self.logger.info(
                f"Shoe type selected: {result.shoe_type.value}, "
                f"safe depth={result.safe_puddle_depth_cm:.1f}cm"
            )
            
            # Automatically return to main screen after 1.5 seconds
            self.logger.info("Auto-returning to main screen after shoe selection")
            Clock.schedule_once(lambda dt: self._return_to_main(), 1.5)
            
        except ValueError:
            self.logger.error(f"Invalid shoe type: {shoe_type_str}")
    
    def on_confirm(self):
        """User confirmed the shoe selection."""
        if self._selected_result is None:
            self.logger.warning("Confirm pressed but no shoe type selected")
            return
        
        # Stop previous speech before announcing confirmation
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("鞋型选择已确认，正在返回主界面")
        
        # Update config with thresholds
        if self.app:
            self.app.app_config.update_puddle_thresholds(
                self._selected_result.puddle_warning_threshold,
                self._selected_result.puddle_critical_threshold
            )
        
        # Save last shoe info
        shoe_type_name = SHOE_TYPE_NAMES.get(self._selected_result.shoe_type.value, "未知")
        self.config.update_last_shoe_info(shoe_type_name, self._selected_result.description)
        
        self.logger.info("Shoe selection confirmed, automatically returning to main screen")
        
        # Automatically return to main screen after 1.5 seconds
        Clock.schedule_once(lambda dt: self._return_to_main(), 1.5)
    
    def _save_selection_and_return(self):
        """Save the selected shoe type and return to entry choice screen."""
        if self._selected_result is None:
            return
        
        # Update config with thresholds
        if self.app:
            self.app.app_config.update_puddle_thresholds(
                self._selected_result.puddle_warning_threshold,
                self._selected_result.puddle_critical_threshold
            )
        
        # Save last shoe info
        shoe_type_name = SHOE_TYPE_NAMES.get(self._selected_result.shoe_type.value, "未知")
        self.config.update_last_shoe_info(shoe_type_name, self._selected_result.description)
        
        self.logger.info("Shoe selection saved, returning to entry choice screen")
        
        # Return to shoe entry choice screen
        if self.app:
            self.app.screen_manager.current = 'shoe_entry_choice'
    
    def _return_to_main(self):
        """Return to main screen."""
        self.logger.info("Returning to main screen from shoe type choice")
        
        # Switch to main screen
        if self.app:
            self.app.screen_manager.current = 'main'
            self.app.state_manager.set_state(AppState.IDLE)
    
    def on_exit(self):
        """User pressed exit button."""
        self.logger.info("Exiting shoe type choice, returning to entry choice")
        
        # Stop previous speech before announcing exit
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在返回鞋型录入选择界面")
        
        # Return to shoe entry choice screen
        if self.app:
            self.app.screen_manager.current = 'shoe_entry_choice'
