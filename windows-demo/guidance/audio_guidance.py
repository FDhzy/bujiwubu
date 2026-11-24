"""
Audio guidance system using Text-to-Speech.
Provides voice feedback for navigation instructions.
使用 Edge TTS 作为默认语音引擎
"""
import threading
import time
from queue import Queue, Empty
from typing import Dict, Optional
from dataclasses import dataclass
from enum import IntEnum

from utils.logger import get_logger
from utils.config import AudioConfig

# Try to import Edge TTS
try:
    from guidance.edge_tts import EdgeTTSWrapper
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class Priority(IntEnum):
    """Audio message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AudioMessage:
    """Message to be spoken."""
    text: str
    priority: Priority
    timestamp: float
    key: str  # For deduplication


class AudioGuidance:
    """
    Provides audio feedback via Text-to-Speech.
    Uses Edge TTS (Microsoft Edge online speech synthesis).
    """
    
    def __init__(self, config: AudioConfig, language: str = "en"):
        """
        Initialize audio guidance system.
        
        Args:
            config: Audio configuration
            language: Language code ("en" or "zh")
        """
        self.logger = get_logger("audio")
        self.config = config
        self.language = language
        
        # Get phrases in current language
        self.phrases = config.phrases if language == "en" else config.phrases_zh
        
        # Message queue
        self._queue: Queue = Queue()
        self._is_running = False
        self._stop_requested = False
        self._speak_thread: Optional[threading.Thread] = None
        
        # Rate limiting
        self._last_message_time: Dict[str, float] = {}
        self._min_interval = config.min_message_interval
        
        # Initialize TTS engine
        self._edge_tts = None
        self._use_edge_tts = False
        self._init_tts()
        
        self.logger.info(f"AudioGuidance initialized (language={language})")
    
    def _init_tts(self):
        """Initialize TTS engine."""
        if EDGE_TTS_AVAILABLE:
            # 使用 Edge TTS（高质量在线语音）
            self.logger.info("Initializing Edge TTS...")
            self._edge_tts = EdgeTTSWrapper(self.config)
            if self._edge_tts.is_available():
                self._use_edge_tts = True
                self.logger.info("Edge TTS initialized successfully")
            else:
                self.logger.warning("Edge TTS not available")
        else:
            self.logger.warning("Edge TTS not installed. Run: pip install edge-tts")
    
    def start(self):
        """Start the audio playback thread."""
        if self._is_running:
            return
        
        self._is_running = True
        self._speak_thread = threading.Thread(
            target=self._speak_loop,
            daemon=True,
            name="AudioGuidance"
        )
        self._speak_thread.start()
        self.logger.info("Audio guidance started")
    
    def stop_current_speech(self):
        """Stop current speech and clear all queued messages without stopping the thread."""
        self.logger.info("Interrupting current speech and clearing queue...")
        
        # Set stop flag for the thread to see
        self._stop_requested = True
        
        # Clear the queue
        cleared_count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared_count += 1
            except Empty:
                break
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} messages from queue")
            
        # Try to stop Edge TTS
        if self._use_edge_tts and self._edge_tts:
            try:
                self._edge_tts.stop()
            except:
                pass
    
    def stop(self):
        """Stop the audio playback thread and clear all queued messages."""
        self.logger.info("Stopping audio guidance - clearing queue and stopping speech...")
        
        # Set flag to stop thread
        self._is_running = False
        
        # Clear all queued messages
        cleared_count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared_count += 1
            except Empty:
                break
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} queued messages")
        
        # Wait for thread to finish (with short timeout)
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=1.0)
            if self._speak_thread.is_alive():
                self.logger.warning("Audio thread did not stop in time")
            else:
                self.logger.info("Audio thread stopped successfully")
    
    def _speak_loop(self):
        """Background thread for processing speech queue."""
        while self._is_running:
            try:
                message = self._queue.get(timeout=0.5)
                # 使用 Edge TTS（在线高质量语音）
                if self._use_edge_tts and self._edge_tts:
                    self._speak_edge_tts(message)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Speak loop error: {e}")
    
    def speak(self, key: str, priority: Priority = Priority.NORMAL):
        """
        Queue a message to be spoken.
        
        Args:
            key: Key into phrases dictionary
            priority: Message priority
        """
        text = self.phrases.get(key, key)
        self.queue_text(text, priority, key)
    
    def queue_text(
        self,
        text: str,
        priority: Priority = Priority.NORMAL,
        key: Optional[str] = None
    ):
        """
        Queue arbitrary text to be spoken.
        
        Args:
            text: Text to speak
            priority: Message priority
            key: Optional key for rate limiting
        """
        # Ensure audio thread is running
        if not self._is_running:
            self.logger.warning("Audio thread not running, starting it now...")
            self.start()
        
        current_time = time.time()
        msg_key = key or text
        
        # Rate limiting
        if msg_key in self._last_message_time:
            elapsed = current_time - self._last_message_time[msg_key]
            if elapsed < self._min_interval and priority < Priority.CRITICAL:
                return
        
        self._last_message_time[msg_key] = current_time
        
        message = AudioMessage(
            text=text,
            priority=priority,
            timestamp=current_time,
            key=msg_key
        )
        
        self._queue.put(message)
        self.logger.debug(f"Queued message: {text} (priority={priority})")
    
    def _speak_edge_tts(self, message: AudioMessage):
        """Speak using Edge TTS."""
        try:
            self.logger.info(f"Speaking (Edge TTS): {message.text}")
            self._stop_requested = False
            
            # Edge TTS 是阻塞播放的，可以直接调用
            self._edge_tts.speak(message.text, block=True)
            
            if not self._stop_requested:
                self.logger.info(f"Finished speaking (Edge TTS): {message.text}")
        except Exception as e:
            self.logger.error(f"Edge TTS speak error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def get_available_voices(self):
        """
        获取可用的音色列表
        
        Returns:
            List[Dict]: 音色列表，每个音色包含 id 和 name
        """
        voices = []
        
        # 返回 Edge TTS 音色
        if self._use_edge_tts and self._edge_tts:
            try:
                edge_voices = self._edge_tts.get_available_voices()
                for voice in edge_voices:
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'description': voice.description
                    })
                return voices
            except Exception as e:
                self.logger.error(f"Failed to get Edge TTS voices: {e}")
        
        return voices
    
    def set_voice(self, voice_type: str):
        """
        设置TTS音色
        
        Args:
            voice_type: 音色类型 ('female', 'male', 'female2', 'male2', 'female3', 'male3', 
                                  'zh_female_1', 'zh_male_1', 'zh_female_2', 'zh_male_2', 
                                  'zh_female_3', 'zh_male_3')
                                  或 Edge TTS 音色ID (如 'zh-CN-XiaoxiaoNeural')
        """
        self.logger.info(f"Setting voice to: {voice_type}")
        
        try:
            # 使用 Edge TTS
            if self._use_edge_tts and self._edge_tts:
                # Edge TTS 音色映射
                edge_voice_mapping = {
                    'female': 'zh-CN-XiaoxiaoNeural',
                    'male': 'zh-CN-YunjianNeural',
                    'female2': 'zh-CN-XiaoyiNeural',
                    'male2': 'zh-CN-YunxiNeural',
                    'female3': 'zh-CN-liaoning-XiaobeiNeural',
                    'male3': 'zh-CN-YunyangNeural'
                }
                edge_voice = edge_voice_mapping.get(voice_type, voice_type)
                self._edge_tts.set_voice(edge_voice)
                self.logger.info(f"Edge TTS voice set to: {edge_voice}")
            else:
                self.logger.warning("Edge TTS not available for voice switching")
                
        except Exception as e:
            self.logger.error(f"Failed to set voice: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def speak_direction(self, direction: str):
        """Speak navigation direction."""
        direction_map = {
            "straight": "go_straight",
            "slight_left": "turn_slight_left",
            "hard_left": "turn_left",
            "slight_right": "turn_slight_right",
            "hard_right": "turn_right",
        }
        key = direction_map.get(direction, "go_straight")
        self.speak(key, Priority.NORMAL)
    
    def speak_warning(self, warning_type: str):
        """Speak warning message."""
        priority = Priority.HIGH if warning_type != "clear" else Priority.LOW
        if warning_type == "stop":
            priority = Priority.CRITICAL
        self.speak(warning_type, priority)
    
    def emergency_stop(self):
        """Speak emergency stop message."""
        self.speak("stop", Priority.CRITICAL)
    
    def release(self):
        """Release TTS resources."""
        self.stop()
        
        if self._edge_tts:
            try:
                self._edge_tts.stop()
            except:
                pass
            self._edge_tts = None
