"""
Edge TTS 集成模块
使用微软 Edge 在线语音合成服务
提供高质量中文语音
优化版本：使用内存缓存和流式播放减少延迟
"""
import os
import io
import tempfile
import asyncio
import threading
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import OrderedDict

try:
    import edge_tts
    import soundfile as sf
    import sounddevice as sd
    import numpy as np
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

from utils.logger import get_logger


@dataclass
class VoiceOption:
    """音色选项"""
    id: str
    name: str
    description: str


class EdgeTTSWrapper:
    """
    Edge TTS 引擎封装（优化版本）
    使用微软 Edge 在线语音合成服务
    支持内存缓存和预合成，减少延迟
    """
    
    # Edge TTS 预定义音色列表（中文）
    CHINESE_VOICES = [
        VoiceOption("zh-CN-XiaoxiaoNeural", "中文女声-晓晓", "活泼女声 - 适合导航引导"),
        VoiceOption("zh-CN-XiaoyiNeural", "中文女声-晓伊", "温柔女声 - 适合提醒提示"),
        VoiceOption("zh-CN-YunjianNeural", "中文男声-云健", "稳重男声 - 适合导航引导"),
        VoiceOption("zh-CN-YunxiNeural", "中文男声-云希", "年轻男声 - 适合轻松氛围"),
        VoiceOption("zh-CN-YunxiaNeural", "中文男声-云夏", "活泼男声 - 适合亲切交流"),
        VoiceOption("zh-CN-YunyangNeural", "中文男声-云扬", "磁性男声 - 适合正式场合"),
        VoiceOption("zh-CN-liaoning-XiaobeiNeural", "中文女声-晓北", "东北方言女声"),
        VoiceOption("zh-CN-shaanxi-XiaoniNeural", "中文女声-晓妮", "陕西方言女声"),
    ]
    
    # 常用提示语预合成缓存
    COMMON_PHRASES = {
        "准备就绪": "准备就绪。请将摄像头向前",
        "欢迎": "欢迎来到主界面",
        "声音设置": "欢迎来到声音设置界面",
        "已切换": "已切换",
    }
    
    def __init__(self, config=None):
        """
        初始化 Edge TTS
        
        Args:
            config: 音频配置对象
        """
        self.logger = get_logger("edge_tts")
        self.config = config
        self.current_voice = "zh-CN-XiaoxiaoNeural"  # 默认使用晓晓
        self._is_speaking = False
        self._stop_requested = False
        self._loop = None
        self._thread = None
        
        # 音频缓存 (LRU Cache)
        self._audio_cache: OrderedDict[str, Tuple[np.ndarray, int]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 20  # 最大缓存数量
        
        # 预合成队列
        self._prewarm_queue = []
        self._prewarm_thread = None
        
        if not EDGE_TTS_AVAILABLE:
            self.logger.warning("Edge TTS not installed. Run: pip install edge-tts soundfile sounddevice numpy")
            return
        
        self.logger.info("Edge TTS initialized (optimized)")
        
        # 启动预合成线程
        self._start_prewarm_thread()
    
    def _start_prewarm_thread(self):
        """启动预合成线程，预合成常用提示语"""
        def prewarm():
            time.sleep(2)  # 等待应用启动完成
            for key, text in self.COMMON_PHRASES.items():
                if not EDGE_TTS_AVAILABLE:
                    return
                try:
                    self._synthesize_to_cache(text)
                    self.logger.debug(f"Prewarmed: {key}")
                except Exception as e:
                    self.logger.debug(f"Failed to prewarm {key}: {e}")
        
        self._prewarm_thread = threading.Thread(target=prewarm, daemon=True)
        self._prewarm_thread.start()
    
    def is_available(self) -> bool:
        """检查TTS是否可用"""
        return EDGE_TTS_AVAILABLE
    
    def get_available_voices(self) -> List[VoiceOption]:
        """获取可用音色列表"""
        return self.CHINESE_VOICES
    
    def set_voice(self, voice_id: str):
        """设置当前音色"""
        if self.current_voice != voice_id:
            self.current_voice = voice_id
            # 切换音色时清空缓存
            with self._cache_lock:
                self._audio_cache.clear()
            self.logger.info(f"Voice set to: {voice_id}, cache cleared")
    
    def set_speech_rate(self, rate: float):
        """设置语速 (0.5 - 2.0)"""
        pass  # Edge TTS 在合成时调整语速
    
    def set_volume(self, volume: float):
        """设置音量 (0.0 - 1.0)"""
        pass  # Edge TTS 在播放时调整音量
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return f"{self.current_voice}:{text}"
    
    def _get_from_cache(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """从缓存获取音频数据"""
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            if cache_key in self._audio_cache:
                # 移动到末尾（最近使用）
                data = self._audio_cache.pop(cache_key)
                self._audio_cache[cache_key] = data
                return data
        return None
    
    def _add_to_cache(self, text: str, audio_data: np.ndarray, samplerate: int):
        """添加音频到缓存"""
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            # 如果缓存已满，删除最旧的
            if len(self._audio_cache) >= self._max_cache_size:
                self._audio_cache.popitem(last=False)
            self._audio_cache[cache_key] = (audio_data, samplerate)
    
    def _synthesize_to_cache(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """合成语音并缓存"""
        try:
            # 检查缓存
            cached = self._get_from_cache(text)
            if cached is not None:
                return cached
            
            # 使用临时文件合成（Edge TTS 不支持 BytesIO）
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                async def synthesize():
                    communicate = edge_tts.Communicate(text, self.current_voice)
                    await communicate.save(tmp_path)
                
                asyncio.run(synthesize())
                
                # 读取音频数据
                audio_data, samplerate = sf.read(tmp_path, dtype='float32')
                
                # 缓存音频数据
                self._add_to_cache(text, audio_data, samplerate)
                
                return audio_data, samplerate
            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            return None
    
    def speak(self, text: str, block: bool = False):
        """
        合成并播放语音（优化版本）
        
        Args:
            text: 要合成的文本
            block: 是否阻塞等待播放完成
        """
        if not EDGE_TTS_AVAILABLE:
            self.logger.warning("Edge TTS not available")
            return
        
        self._stop_requested = False
        
        if block:
            self._do_speak(text)
        else:
            # 在后台线程中播放
            thread = threading.Thread(target=self._do_speak, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _do_speak(self, text: str):
        """执行语音合成和播放（优化版本）"""
        try:
            self._is_speaking = True
            start_time = time.time()
            
            # 尝试从缓存获取
            audio_data = self._get_from_cache(text)
            
            if audio_data is None:
                # 缓存未命中，实时合成
                self.logger.debug(f"Cache miss for: {text[:30]}...")
                audio_data = self._synthesize_to_cache(text)
            else:
                self.logger.debug(f"Cache hit for: {text[:30]}...")
            
            if audio_data is None:
                self.logger.error(f"Failed to synthesize: {text[:50]}")
                return
            
            data, samplerate = audio_data
            synthesis_time = time.time() - start_time
            
            # 播放语音
            if not self._stop_requested:
                sd.play(data, samplerate)
                sd.wait()
            
            total_time = time.time() - start_time
            self.logger.debug(f"Speech completed in {total_time:.2f}s (synthesis: {synthesis_time:.2f}s)")
                
        except Exception as e:
            self.logger.error(f"Edge TTS speak error: {e}")
        finally:
            self._is_speaking = False
    
    async def _synthesize(self, text: str, output_path: str):
        """异步合成语音（保留用于兼容）"""
        communicate = edge_tts.Communicate(text, self.current_voice)
        await communicate.save(output_path)
    
    def stop(self):
        """停止播放"""
        self._stop_requested = True
        if self._is_speaking:
            sd.stop()
        self.logger.info("Edge TTS stopped")
    
    def is_speaking(self) -> bool:
        """检查是否正在播放"""
        return self._is_speaking
