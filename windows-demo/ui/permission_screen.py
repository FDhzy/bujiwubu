from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from utils.logger import get_logger

class PermissionScreen(Screen):
    """Screen for explaining application permissions to the user."""
    
    app = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        self.logger = get_logger("ui")
        super().__init__(**kwargs)
    
    def on_enter(self):
        """Called when the screen is displayed."""
        if self.app and self.app.feedback_controller:
            permission_text = (
                "权限及隐私说明。为了为您提供更安全、精准的避障服务，“避障助手”需要获取以下权限： "
                "第一，相机权限，用于检测前方路面的水坑和障碍物。 "
                "第二，麦克风权限，用于语音指令识别和唤醒功能。 "
                "第三，存储权限，用于保存您的个人偏好和鞋型设置。 "
                "第四，震动权限，用于在紧急情况下向您发出警报。 "
                "我们非常重视您的隐私，所有识别过程均在本地完成，不会上传您的数据。 "
                "点击屏幕下方的确认按钮表示您同意上述权限及隐私政策。"
            )
            self.app.feedback_controller.speak_text(permission_text)

    def on_confirm(self):
        """Handle 'Confirm' button press."""
        self.logger.info("Permissions confirmed by user")
        if self.app:
            # Stop the permission announcement
            if self.app.feedback_controller:
                self.app.feedback_controller.stop_audio()
                
            # Mark first run as complete and save
            self.app.app_config.is_first_run = False
            self.app.app_config._save_shoe_info()
            
            # Navigate to voice settings screen
            self.app.screen_manager.current = 'voice_settings'
