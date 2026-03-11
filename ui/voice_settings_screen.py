"""
声音设置界面 - 调整音色、语速、音量
"""
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty
from utils.logger import get_logger


class VoiceSettingsScreen(Screen):
    """声音设置界面"""
    
    app = ObjectProperty(None)
    current_voice = StringProperty('female')  # 默认女声
    speech_rate = NumericProperty(130)  # 默认语速130%
    volume = NumericProperty(0.8)  # 默认音量80%
    
    # 可用音色列表（形象化名称）
    available_voices = ListProperty([
        ('female', '晓晓'),      # 活泼女声
        ('male', '云健'),        # 稳重男声
        ('female2', '晓伊'),     # 温柔女声
        ('male2', '云希'),       # 年轻男声
        ('female3', '晓北'),     # 东北女声
        ('male3', '云扬')        # 磁性男声
    ])
    
    def __init__(self, **kwargs):
        self.logger = get_logger("ui")
        super().__init__(**kwargs)
    
    def on_enter(self):
        """进入界面时加载当前设置"""
        self.logger.info("Entering voice settings screen")
        
        # 从配置加载当前设置
        if self.app and self.app.app_config:
            config = self.app.app_config
            self.speech_rate = getattr(config, 'speech_rate', 130)
            self.volume = getattr(config, 'speech_volume', 0.8)
            self.current_voice = getattr(config, 'voice_type', 'female')
        
        # 动态更新可用音色列表
        self._update_voice_buttons()
        
        # 播报界面说明（简洁版，减少延迟）
        if self.app and self.app.feedback_controller:
            intro_text = "声音设置。滑动调节语速音量，点击按钮切换音色。"
            self.app.feedback_controller.speak_text(intro_text)
    
    def _update_voice_buttons(self):
        """根据可用的TTS引擎更新音色按钮"""
        if not self.app or not self.app.feedback_controller:
            return
        
        # 获取音频引导对象
        audio = self.app.feedback_controller.audio
        if not audio:
            return
        
        # 获取可用音色列表
        available_voices = audio.get_available_voices()
        self.logger.info(f"Available voices: {len(available_voices)}")
        
        # 更新UI显示（这里可以根据可用音色动态调整按钮）
        # 暂时只记录日志，UI保持固定6个按钮
        for voice in available_voices:
            self.logger.debug(f"Voice: {voice['id']} - {voice['name']}")
    
    def on_voice_type_selected(self, voice_type: str):
        """选择音色并播报项目介绍"""
        self.current_voice = voice_type
        self.logger.info(f"Voice type selected: {voice_type}")
        
        # 停止当前正在播放的语音
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
        
        # 立即应用音色设置到TTS引擎
        self._apply_voice_setting(voice_type)
        
        # 播报项目介绍（展示音色效果）
        if self.app and self.app.feedback_controller:
            # 形象化音色名称（带特点描述）
            voice_names = {
                'female': '晓晓，活泼女声',
                'male': '云健，稳重男声',
                'female2': '晓伊，温柔女声',
                'male2': '云希，年轻男声',
                'female3': '晓北，东北女声',
                'male3': '云扬，磁性男声'
            }
            voice_name = voice_names.get(voice_type, '当前')
            
            # 项目介绍文本
            intro_text = (
                f"已切换至{voice_name}。"
                "欢迎使用不积洼步智能导引系统。"
                "本系统专为视障人士设计，通过实时分析摄像头画面，"
                "帮助您识别前方的水坑、障碍物和鞋子类型。"
                "系统会提供语音导航，告诉您向左或向右走几步来避开危险。"
                "您可以在声音设置中调节语速、音量和音色。"
                "现在点击确认按钮保存设置，开始安全出行。"
            )
            self.app.feedback_controller.speak_text(intro_text)
    
    def _apply_voice_setting(self, voice_type: str):
        """应用音色设置到TTS引擎"""
        if self.app and self.app.feedback_controller:
            try:
                # 获取音频引导对象
                audio_guidance = self.app.feedback_controller.audio
                
                # 调用设置音色的方法
                if hasattr(audio_guidance, 'set_voice'):
                    audio_guidance.set_voice(voice_type)
                    self.logger.info(f"Voice setting applied: {voice_type}")
                else:
                    self.logger.warning("Audio guidance does not support voice switching")
                    
            except Exception as e:
                self.logger.error(f"Failed to apply voice setting: {e}")
    
    def on_speech_rate_change(self, value: float):
        """语速改变"""
        self.speech_rate = value
        self.logger.debug(f"Speech rate changed to: {value}")
    
    def on_volume_change(self, value: float):
        """音量改变"""
        self.volume = value / 100.0  # 转换为0-1范围
        self.logger.debug(f"Volume changed to: {self.volume}")
    
    def on_test_voice(self):
        """试听当前设置"""
        self.logger.info("Testing voice settings")
        
        if self.app and self.app.feedback_controller:
            # 临时应用设置进行试听
            self._apply_settings()
            
            test_text = (
                f"当前语速为{int(self.speech_rate)}%，"
                f"音量为{int(self.volume * 100)}%，"
                f"音色为{'女声' if self.current_voice == 'female' else '男声'}。"
                "这是测试语音效果。"
            )
            self.app.feedback_controller.speak_text(test_text)
    
    def on_confirm(self):
        """确认保存设置"""
        self.logger.info("Voice settings confirmed")
        
        # 停止当前正在播放的语音
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
        
        # 保存设置到配置
        self._apply_settings()
        
        if self.app and self.app.app_config:
            self.app.app_config._save_shoe_info()
        
        # 播放确认提示
        if self.app and self.app.feedback_controller:
            self.app.feedback_controller.speak_text("声音设置已保存")
        
        # 跳转到主界面
        if self.app:
            self.app.screen_manager.current = 'main'
    
    def _apply_settings(self):
        """应用设置到反馈控制器"""
        if self.app and self.app.app_config:
            config = self.app.app_config
            config.speech_rate = self.speech_rate / 100.0  # 转换为倍率
            config.speech_volume = self.volume
            config.voice_type = self.current_voice
            
            # 更新反馈控制器
            if self.app.feedback_controller:
                # 这里可以添加实际的TTS引擎参数更新
                pass
