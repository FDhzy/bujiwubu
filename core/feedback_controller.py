"""
Feedback controller coordinating audio and haptic output.
Translates navigation results into user-facing feedback.
"""
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.config import AudioConfig, HapticConfig
from guidance.audio_guidance import AudioGuidance, Priority
from guidance.haptic_patterns import HapticFeedback, HapticPattern
from core.path_planner import NavigationResult


class FeedbackState(Enum):
    """Current feedback state."""
    IDLE = "idle"
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FeedbackAction:
    """Feedback action to take."""
    speak_key: Optional[str]
    haptic_pattern: Optional[HapticPattern]
    priority: Priority
    state: FeedbackState


class FeedbackController:
    """
    Coordinates audio and haptic feedback for navigation guidance.
    Manages feedback timing, prioritization, and user output.
    """
    
    def __init__(
        self,
        audio_config: AudioConfig,
        haptic_config: HapticConfig,
        language: str = "zh"
    ):
        """
        Initialize feedback controller.
        
        Args:
            audio_config: Audio guidance configuration
            haptic_config: Haptic feedback configuration
            language: Language for audio ("en" or "zh")
        """
        self.logger = get_logger("feedback")
        
        # Initialize subsystems
        self.audio = AudioGuidance(audio_config, language)
        self.haptic = HapticFeedback(haptic_config)
        
        # State tracking
        self._current_state = FeedbackState.IDLE
        self._last_direction = "straight"
        self._last_warning_time = 0.0
        self._last_direction_time = 0.0
        
        # Timing configuration
        self._direction_interval = 3.0  # Seconds between direction announcements
        self._warning_interval = 2.0    # Seconds between warnings
        
        # Start audio thread
        self.audio.start()
        
        self.logger.info("FeedbackController initialized")
    
    def process_navigation(self, navigation: NavigationResult):
        """
        Process navigation result and provide appropriate feedback.
        
        Args:
            navigation: Navigation result from path planner
        """
        current_time = time.time()
        
        # Determine feedback action based on navigation result
        action = self._determine_action(navigation)
        
        # Update state
        old_state = self._current_state
        self._current_state = action.state
        
        # Execute feedback
        self._execute_feedback(action, navigation, current_time)
        
        # Log state change
        if old_state != self._current_state:
            self.logger.debug(f"Feedback state: {old_state.value} -> {self._current_state.value}")
    
    def _determine_action(self, nav: NavigationResult) -> FeedbackAction:
        """Determine what feedback action to take based on navigation and shoe thresholds."""
        # Critical situation - immediate stop
        if nav.should_stop:
            return FeedbackAction(
                speak_key="stop",
                haptic_pattern=HapticPattern.CRITICAL_ALERT,
                priority=Priority.CRITICAL,
                state=FeedbackState.CRITICAL
            )
        
        # Get puddle severity (uses shoe-based thresholds)
        puddle_severity = getattr(nav, 'puddle_severity', 'safe')
        
        # Warning situation
        if nav.safety_score < 0.5:
            if nav.hazard_type == "puddle":
                # Use shoe-aware puddle warnings
                if puddle_severity == "critical":
                    return FeedbackAction(
                        speak_key="puddle_deep",
                        haptic_pattern=HapticPattern.CRITICAL_ALERT,
                        priority=Priority.CRITICAL,
                        state=FeedbackState.WARNING
                    )
                elif puddle_severity == "warning":
                    return FeedbackAction(
                        speak_key="puddle_ahead",
                        haptic_pattern=HapticPattern.PUDDLE_WARNING,
                        priority=Priority.HIGH,
                        state=FeedbackState.WARNING
                    )
                else:
                    # Shallow puddle - safe for current shoes
                    return FeedbackAction(
                        speak_key="puddle_shallow",
                        haptic_pattern=HapticPattern.SUCCESS,
                        priority=Priority.NORMAL,
                        state=FeedbackState.CAUTION
                    )
            elif nav.hazard_type == "obstacle":
                return FeedbackAction(
                    speak_key="obstacle_ahead",
                    haptic_pattern=HapticPattern.OBSTACLE_WARNING,
                    priority=Priority.HIGH,
                    state=FeedbackState.WARNING
                )
            else:
                return FeedbackAction(
                    speak_key="caution",
                    haptic_pattern=HapticPattern.PUDDLE_WARNING,
                    priority=Priority.HIGH,
                    state=FeedbackState.WARNING
                )
        
        # Caution situation - also check puddle severity
        if nav.safety_score < 0.7:
            if nav.hazard_type == "puddle" and puddle_severity == "safe":
                # Puddle is safe for current shoes, just give direction
                return FeedbackAction(
                    speak_key="puddle_shallow",
                    haptic_pattern=HapticPattern.SUCCESS,
                    priority=Priority.LOW,
                    state=FeedbackState.NORMAL
                )
            return FeedbackAction(
                speak_key=self._direction_to_key(nav.direction),
                haptic_pattern=HapticPattern.DIRECTION_CHANGE if nav.direction != "straight" else None,
                priority=Priority.NORMAL,
                state=FeedbackState.CAUTION
            )
        
        # Normal situation
        speak_key = None
        if nav.direction != self._last_direction:
            speak_key = self._direction_to_key(nav.direction)
        
        return FeedbackAction(
            speak_key=speak_key,
            haptic_pattern=HapticPattern.SUCCESS if nav.direction == "straight" else HapticPattern.DIRECTION_CHANGE,
            priority=Priority.LOW,
            state=FeedbackState.NORMAL
        )
    
    def _direction_to_key(self, direction: str) -> str:
        """Convert direction name to audio phrase key."""
        direction_map = {
            "straight": "go_straight",
            "slight_left": "turn_slight_left",
            "hard_left": "turn_left",
            "slight_right": "turn_slight_right",
            "hard_right": "turn_right",
        }
        return direction_map.get(direction, "go_straight")
    
    def _execute_feedback(
        self,
        action: FeedbackAction,
        nav: NavigationResult,
        current_time: float
    ):
        """Execute feedback action with rate limiting."""
        # Handle audio
        if action.speak_key:
            if action.state in (FeedbackState.CRITICAL, FeedbackState.WARNING):
                # Always speak warnings/critical immediately
                if current_time - self._last_warning_time > self._warning_interval or \
                   action.priority >= Priority.CRITICAL:
                    self.audio.speak(action.speak_key, action.priority)
                    self._last_warning_time = current_time
            else:
                # Rate limit direction announcements
                direction_changed = nav.direction != self._last_direction
                time_elapsed = current_time - self._last_direction_time > self._direction_interval
                
                if direction_changed or time_elapsed:
                    self.audio.speak(action.speak_key, action.priority)
                    self._last_direction_time = current_time
                    self._last_direction = nav.direction
        
        # Handle haptic
        if action.haptic_pattern:
            # Critical haptic always triggers
            if action.state == FeedbackState.CRITICAL:
                self.haptic.vibrate(action.haptic_pattern)
            # Other haptics are rate limited by the HapticFeedback class
            elif action.state == FeedbackState.WARNING:
                self.haptic.vibrate(action.haptic_pattern)
            elif nav.direction != self._last_direction:
                self.haptic.vibrate(action.haptic_pattern)
    
    def speak(self, key: str, priority: Priority = Priority.NORMAL):
        """
        Speak a message by key.
        
        Args:
            key: Phrase key from config
            priority: Message priority
        """
        self.audio.speak(key, priority)
    
    def speak_text(self, text: str, priority: Priority = Priority.NORMAL):
        """
        Speak arbitrary text.
        
        Args:
            text: Text to speak
            priority: Message priority
        """
        self.audio.queue_text(text, priority)
    
    def vibrate(self, pattern: HapticPattern):
        """
        Trigger a haptic pattern.
        
        Args:
            pattern: Pattern to vibrate
        """
        self.haptic.vibrate(pattern)
    
    def stop_audio(self):
        """Stop current speech and clear the audio queue."""
        self.audio.stop_current_speech()
    
    def emergency_stop(self):
        """Trigger emergency stop feedback."""
        self._current_state = FeedbackState.CRITICAL
        self.audio.emergency_stop()
        self.haptic.critical_alert()
    
    def announce_clear(self):
        """Announce that path is clear."""
        self.audio.speak("clear", Priority.LOW)
        self.haptic.success_feedback()
    
    def announce_ready(self):
        """Announce that system is ready."""
        self.audio.speak("ready", Priority.NORMAL)
    
    def set_direction_interval(self, seconds: float):
        """Set minimum interval between direction announcements."""
        self._direction_interval = max(1.0, seconds)
    
    def set_warning_interval(self, seconds: float):
        """Set minimum interval between warning announcements."""
        self._warning_interval = max(0.5, seconds)
    
    def process_precise_guidance(self, instruction, language: str = "zh"):
        """
        Process precise guidance instruction and provide feedback.
        
        Args:
            instruction: PreciseInstruction from precise_guidance module
            language: Language for TTS ("en" or "zh")
        """
        current_time = time.time()
        
        # Get text in appropriate language
        text = instruction.text_zh if language == "zh" else instruction.text_en
        
        # Determine priority and haptic pattern
        if instruction.priority >= 4:
            priority = Priority.CRITICAL
            haptic_pattern = HapticPattern.CRITICAL_ALERT
            state = FeedbackState.CRITICAL
        elif instruction.priority >= 3:
            priority = Priority.HIGH
            haptic_pattern = HapticPattern.OBSTACLE_WARNING if instruction.instruction_type == "obstacle" else HapticPattern.PUDDLE_WARNING
            state = FeedbackState.WARNING
        elif instruction.priority >= 2:
            priority = Priority.NORMAL
            haptic_pattern = HapticPattern.DIRECTION_CHANGE
            state = FeedbackState.CAUTION
        else:
            priority = Priority.LOW
            haptic_pattern = HapticPattern.SUCCESS
            state = FeedbackState.NORMAL
        
        # Rate limiting for precise guidance
        if state in (FeedbackState.CRITICAL, FeedbackState.WARNING):
            if current_time - self._last_warning_time > self._warning_interval or priority >= Priority.CRITICAL:
                self.audio.queue_text(text, priority)
                self.haptic.vibrate(haptic_pattern)
                self._last_warning_time = current_time
        else:
            if current_time - self._last_direction_time > self._direction_interval:
                self.audio.queue_text(text, priority)
                if haptic_pattern:
                    self.haptic.vibrate(haptic_pattern)
                self._last_direction_time = current_time
        
        self._current_state = state
    
    def release(self):
        """Release all resources."""
        self.logger.info("Releasing feedback controller")
        
        self.audio.release()
        self.haptic.release()
        
        self.logger.info("Feedback controller released")
