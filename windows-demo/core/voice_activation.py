"""
Voice activation module for wake word detection.
Listens for "拜拜，雨坑" (Bye bye, puddle) to start the app.
"""
import threading
import time
from typing import Callable, Optional, List
from enum import Enum

from utils.logger import get_logger

# Try to import speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Check for Android
try:
    from kivy.utils import platform
    if platform == 'android':
        from jnius import autoclass
        ANDROID = True
    else:
        ANDROID = False
except (ImportError, Exception):
    ANDROID = False


class VoiceActivation:
    """
    Listens for wake word to activate the navigation assistant.
    Supports both desktop (speech_recognition) and Android platforms.
    """
    
    # Wake words in different languages
    WAKE_WORDS = [
        "下雨了，我要出门",
    ]
    
    # Command words for control
    COMMANDS = {
        "stop": ["停止", "暂停", "stop", "pause"],
        "start": ["开始", "继续", "start", "resume", "go"],
        "repeat": ["重复", "再说一遍", "repeat", "again"],
        "help": ["帮助", "help"],
        "reach": ["我已到达目的地", "到达目的地", "arrived", "reached destination"],
    }
    
    def __init__(
        self,
        on_wake: Optional[Callable[[], None]] = None,
        on_command: Optional[Callable[[str], None]] = None,
        language: str = "zh-CN"
    ):
        """
        Initialize voice activation.
        
        Args:
            on_wake: Callback when wake word detected
            on_command: Callback when command detected
            language: Recognition language
        """
        self.logger = get_logger("voice")
        self.on_wake = on_wake
        self.on_command = on_command
        self.language = language
        
        self._is_listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._recognizer = None
        self._microphone = None
        
        # Initialize speech recognition
        self._init_recognition()
        
        self.logger.info("VoiceActivation initialized")
    
    def _init_recognition(self):
        """Initialize speech recognition engine."""
        if SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            # 设置较低的能量阈值以提高敏感度
            self._recognizer.energy_threshold = 200
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8
            
            try:
                self._microphone = sr.Microphone()
                # Calibrate for ambient noise
                with self._microphone as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Speech recognition initialized (desktop)")
            except Exception as e:
                self.logger.warning(f"Microphone init failed: {e}")
                self._microphone = None
        
        elif ANDROID:
            self._init_android_recognition()
        else:
            self.logger.warning("No speech recognition available")
        
        # 添加一个标志位，指示语音激活是否可用
        self._available = SR_AVAILABLE and self._microphone is not None
        if not self._available:
            self.logger.warning("Voice activation is not available")
    
    def _init_android_recognition(self):
        """Initialize Android speech recognition."""
        try:
            SpeechRecognizer = autoclass('android.speech.SpeechRecognizer')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            
            context = PythonActivity.mActivity
            if SpeechRecognizer.isRecognitionAvailable(context):
                self.logger.info("Android speech recognition available")
            else:
                self.logger.warning("Android speech recognition not available")
        except Exception as e:
            self.logger.error(f"Android recognition init failed: {e}")
    
    def start_listening(self):
        """Start listening for wake word in background."""
        if self._is_listening:
            return
        
        if not SR_AVAILABLE or self._microphone is None:
            self.logger.warning("Cannot start listening - no microphone")
            return
        
        self._is_listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="VoiceActivation"
        )
        self._listen_thread.start()
        self.logger.info("Started listening for wake word")
    
    def stop_listening(self):
        """Stop listening for wake word."""
        self._is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        self.logger.info("Stopped listening")
    
    def _listen_loop(self):
        """Background thread for continuous listening."""
        while self._is_listening:
            try:
                with self._microphone as source:
                    self.logger.debug("Listening...")
                    audio = self._recognizer.listen(
                        source, 
                        timeout=3,  # 减少超时时间以提高响应速度
                        phrase_time_limit=5
                    )
                
                # Try to recognize
                text = self._recognize(audio)
                if text:
                    self._process_recognition(text)
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.logger.debug(f"Listen error: {e}")
                time.sleep(0.5)
    
    def _recognize(self, audio) -> Optional[str]:
        """Recognize speech from audio."""
        try:
            # 首先尝试Google Speech Recognition
            try:
                text = self._recognizer.recognize_google(
                    audio, 
                    language=self.language
                )
                self.logger.debug(f"Recognized (Google): {text}")
                return text.lower()
            except sr.RequestError:
                # 如果Google服务不可用，尝试其他方法
                self.logger.debug("Google recognition failed, trying alternative methods")
                
            # 如果有离线识别可用，可以在这里添加备用方案
            # 目前暂时返回None
            return None
            
        except sr.UnknownValueError:
            # 音频无法理解
            return None
        except sr.RequestError as e:
            self.logger.warning(f"Recognition service error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected recognition error: {e}")
            return None
    
    def _process_recognition(self, text: str):
        """Process recognized text for wake word or commands."""
        self.logger.debug(f"Processing recognized text: {text}")
        
        # Clean text: remove common Chinese punctuation and whitespace
        clean_text = text.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip().lower()
        
        # Check for wake word
        for wake_word in self.WAKE_WORDS:
            clean_wake = wake_word.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip().lower()
            if clean_wake in clean_text or clean_text == clean_wake:
                self.logger.info(f"Wake word detected: {wake_word}")
                if self.on_wake:
                    self.on_wake()
                return
        
        # Check for commands
        for command, phrases in self.COMMANDS.items():
            for phrase in phrases:
                clean_phrase = phrase.replace("。", "").replace("，", "").replace("！", "").replace("？", "").strip().lower()
                if clean_phrase in clean_text or clean_text == clean_phrase:
                    self.logger.info(f"Command detected: {command}")
                    if self.on_command:
                        self.on_command(command)
                    return
        
        # 如果没有匹配到任何关键词，记录下来用于调试
        self.logger.debug(f"Unrecognized text: {text}")
    
    def listen_once(self) -> Optional[str]:
        """
        Listen for a single utterance and return recognized text.
        Blocking call.
        
        Returns:
            Recognized text or None
        """
        if not SR_AVAILABLE or self._microphone is None:
            return None
        
        try:
            with self._microphone as source:
                audio = self._recognizer.listen(source, timeout=10)
            return self._recognize(audio)
        except Exception as e:
            self.logger.error(f"Listen once error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if voice recognition is available."""
        return self._available
    
    def release(self):
        """Release resources."""
        self.stop_listening()
