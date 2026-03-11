"""
Shoe entry choice screen - 鞋子录入方式选择界面
"""
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from utils.logger import get_logger
from utils.config import get_config
import os

# Load KV file
kv_path = os.path.join(os.path.dirname(__file__), 'shoe_entry_choice_screen.kv')
Builder.load_file(kv_path)


class ShoeEntryChoiceScreen(Screen):
    """
    Screen for choosing shoe entry method.
    User can select camera capture, gallery selection, or exit.
    """
    
    app = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger("shoe_entry_choice")
        self.config = get_config()
    
    def on_enter(self):
        """Called when screen is displayed."""
        self.logger.info("Entering shoe entry choice screen")
        
        # Update title label with last shoe info
        title_label = self.ids.get('title_label')
        if title_label:
            if self.config.last_shoe_type and self.config.last_shoe_type != "无":
                # 显示鞋型名称和完整的防水描述
                title_label.text = f"鞋子录入\n最近录入：{self.config.last_shoe_type}\n{self.config.last_shoe_description}"
            else:
                title_label.text = "鞋子录入"
        
        # Announce instructions via TTS
        if self.app and self.app.feedback_controller:
            instruction_text = (
                "欢迎来到鞋子录入界面，请选择录入方式。屏幕中心分布着两个大按钮："
                "上方是拍摄录入，下方是选择录入。屏幕底部右侧是退出按钮。"
            )
            self.app.feedback_controller.speak_text(instruction_text)
    
    def on_camera_entry(self):
        """User selected camera capture entry."""
        self.logger.info("User selected camera entry")
        
        # Stop previous speech before announcing selection
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("已选择拍照录入方式")
        
        # Navigate to shoe setup screen (camera capture)
        if self.app:
            self.app.screen_manager.current = 'shoe_setup'
    
    def on_gallery_entry(self):
        """User selected gallery/file selection entry."""
        self.logger.info("User selected gallery entry")
        
        # Stop previous speech before announcing selection
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("已选择图库录入方式")
        
        # Navigate to shoe type choice screen
        if self.app:
            self.app.screen_manager.current = 'shoe_type_choice'
    
    def on_exit(self):
        """User chose to exit shoe entry."""
        self.logger.info("User exited shoe entry choice")
        
        # Stop previous speech before announcing exit
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在返回主界面")
        
        # Return to main screen
        if self.app:
            self.app.screen_manager.current = 'main'
