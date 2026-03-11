"""
Haptic feedback patterns for navigation guidance.
Provides vibration feedback on Android devices.
"""
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.config import HapticConfig

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

# Try plyer for cross-platform vibration
try:
    from plyer import vibrator
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False


class HapticPattern(Enum):
    """Predefined haptic patterns."""
    PUDDLE_WARNING = "puddle_warning"
    OBSTACLE_WARNING = "obstacle_warning"
    CRITICAL_ALERT = "critical_alert"
    SUCCESS = "success"
    DIRECTION_CHANGE = "direction_change"


@dataclass
class VibrationSequence:
    """A sequence of vibration pulses."""
    # List of (vibrate_ms, pause_ms) tuples
    pulses: List[Tuple[int, int]]
    # Total duration in milliseconds
    total_duration: int


class HapticFeedback:
    """
    Provides haptic (vibration) feedback for navigation.
    Supports Android devices with vibration capability.
    """
    
    def __init__(self, config: HapticConfig):
        """
        Initialize haptic feedback system.
        
        Args:
            config: Haptic configuration with pattern definitions
        """
        self.logger = get_logger("haptic")
        self.config = config
        
        # Parse patterns from config
        self.patterns = self._parse_patterns(config.patterns)
        
        # Rate limiting
        self._last_vibrate_time = 0.0
        self._min_interval = config.min_haptic_interval
        
        # Initialize vibrator
        self._vibrator = None
        self._init_vibrator()
        
        self.logger.info("HapticFeedback initialized")
    
    def _parse_patterns(
        self,
        pattern_config: dict
    ) -> dict:
        """Parse pattern configuration into VibrationSequence objects."""
        patterns = {}
        
        for name, pulses in pattern_config.items():
            total = sum(v + p for v, p in pulses)
            patterns[name] = VibrationSequence(
                pulses=pulses,
                total_duration=total
            )
        
        return patterns
    
    def _init_vibrator(self):
        """Initialize platform-specific vibrator."""
        if ANDROID:
            self._init_android_vibrator()
        elif PLYER_AVAILABLE:
            self._vibrator = vibrator
            self.logger.info("Using plyer vibrator")
        else:
            self.logger.warning("No vibration support available")
    
    def _init_android_vibrator(self):
        """Initialize Android Vibrator service."""
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = autoclass('android.content.Context')
            
            context = PythonActivity.mActivity
            self._vibrator = context.getSystemService(Context.VIBRATOR_SERVICE)
            
            if self._vibrator and self._vibrator.hasVibrator():
                self.logger.info("Android Vibrator initialized")
            else:
                self.logger.warning("Device does not have vibrator")
                self._vibrator = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Android vibrator: {e}")
            self._vibrator = None
    
    def vibrate(self, pattern: HapticPattern) -> bool:
        """
        Trigger a predefined vibration pattern.
        
        Args:
            pattern: The pattern to play
            
        Returns:
            True if vibration was triggered, False otherwise
        """
        pattern_name = pattern.value
        
        if pattern_name not in self.patterns:
            self.logger.warning(f"Unknown pattern: {pattern_name}")
            return False
        
        return self.vibrate_sequence(self.patterns[pattern_name])
    
    def vibrate_sequence(self, sequence: VibrationSequence) -> bool:
        """
        Play a vibration sequence.
        
        Args:
            sequence: VibrationSequence to play
            
        Returns:
            True if successful
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self._last_vibrate_time < self._min_interval:
            return False
        
        self._last_vibrate_time = current_time
        
        if ANDROID and self._vibrator is not None:
            return self._vibrate_android(sequence)
        elif PLYER_AVAILABLE and self._vibrator is not None:
            return self._vibrate_plyer(sequence)
        else:
            # Log pattern for debugging
            self.logger.debug(f"Vibration: {sequence.pulses}")
            return False
    
    def _vibrate_android(self, sequence: VibrationSequence) -> bool:
        """Vibrate using Android Vibrator API."""
        try:
            VibrationEffect = autoclass('android.os.VibrationEffect')
            
            # Convert pattern to Android format
            # Android expects: [pause, vibrate, pause, vibrate, ...]
            pattern = []
            for vibrate_ms, pause_ms in sequence.pulses:
                if not pattern:
                    pattern.append(0)  # Initial wait
                pattern.append(vibrate_ms)
                if pause_ms > 0:
                    pattern.append(pause_ms)
            
            # Create long array for Java
            from jnius import cast
            pattern_array = pattern
            
            # Create VibrationEffect (Android 8.0+)
            try:
                effect = VibrationEffect.createWaveform(pattern_array, -1)
                self._vibrator.vibrate(effect)
            except:
                # Fallback for older Android
                self._vibrator.vibrate(pattern_array, -1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Android vibration error: {e}")
            return False
    
    def _vibrate_plyer(self, sequence: VibrationSequence) -> bool:
        """Vibrate using plyer (cross-platform)."""
        try:
            # Plyer only supports simple duration
            total_ms = sequence.total_duration
            self._vibrator.vibrate(time=total_ms / 1000.0)
            return True
        except Exception as e:
            self.logger.error(f"Plyer vibration error: {e}")
            return False
    
    def vibrate_simple(self, duration_ms: int) -> bool:
        """
        Simple vibration for a given duration.
        
        Args:
            duration_ms: Vibration duration in milliseconds
            
        Returns:
            True if successful
        """
        sequence = VibrationSequence(
            pulses=[(duration_ms, 0)],
            total_duration=duration_ms
        )
        return self.vibrate_sequence(sequence)
    
    def puddle_warning(self):
        """Vibrate puddle warning pattern."""
        self.vibrate(HapticPattern.PUDDLE_WARNING)
    
    def obstacle_warning(self):
        """Vibrate obstacle warning pattern."""
        self.vibrate(HapticPattern.OBSTACLE_WARNING)
    
    def critical_alert(self):
        """Vibrate critical alert pattern."""
        self.vibrate(HapticPattern.CRITICAL_ALERT)
    
    def success_feedback(self):
        """Vibrate success pattern."""
        self.vibrate(HapticPattern.SUCCESS)
    
    def direction_change(self):
        """Vibrate direction change pattern."""
        self.vibrate(HapticPattern.DIRECTION_CHANGE)
    
    def cancel(self):
        """Cancel any ongoing vibration."""
        if ANDROID and self._vibrator is not None:
            try:
                self._vibrator.cancel()
            except:
                pass
    
    def release(self):
        """Release vibrator resources."""
        self.cancel()
