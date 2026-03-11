"""
Blind Navigation Assistant - Main Application Entry Point

A real-time camera-based navigation assistant for blind users
that detects water puddles and obstacles, providing audio and
haptic guidance for safe walking in rainy weather.
"""
import os
import sys

# Set environment variables before importing Kivy
os.environ['KIVY_LOG_LEVEL'] = 'info'

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager

from utils.config import get_config, set_debug_mode, AppState
from utils.logger import setup_logger, get_logger
from utils.performance_monitor import get_monitor
from ui.main_screen import MainScreen
from ui.shoe_setup_screen import ShoeSetupScreen
from ui.shoe_entry_choice_screen import ShoeEntryChoiceScreen
from ui.shoe_type_choice_screen import ShoeTypeChoiceScreen
from ui.permission_screen import PermissionScreen
from ui.voice_settings_screen import VoiceSettingsScreen
from core.state_manager import StateManager
from core.camera_manager import CameraManager
from core.inference_engine import InferenceEngine
from core.path_planner import PathPlanner
from core.feedback_controller import FeedbackController
from core.voice_activation import VoiceActivation
from guidance.distance_estimator import DistanceEstimator
from guidance.precise_guidance import PreciseGuidanceGenerator


# Register Chinese font support
# Try common Chinese font paths on Windows
try:
    chinese_fonts = [
        'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei
        'C:/Windows/Fonts/msyhbd.ttc',  # Microsoft YaHei Bold
        'C:/Windows/Fonts/simhei.ttf',  # SimHei
        'C:/Windows/Fonts/simsun.ttc',  # SimSun
    ]
    
    font_found = False
    for font_path in chinese_fonts:
        if os.path.exists(font_path):
            LabelBase.register(
                name='ChineseFont',
                fn_regular=font_path
            )
            font_found = True
            print(f"Registered Chinese font: {font_path}")
            break
    
    if not font_found:
        print("Warning: No Chinese font found. Chinese text may not display correctly.")
except Exception as e:
    print(f"Error registering Chinese font: {e}")

# Load KV files
kv_path = os.path.join(os.path.dirname(__file__), 'ui', 'main_screen.kv')
if os.path.exists(kv_path):
    Builder.load_file(kv_path)

shoe_kv_path = os.path.join(os.path.dirname(__file__), 'ui', 'shoe_setup_screen.kv')
if os.path.exists(shoe_kv_path):
    Builder.load_file(shoe_kv_path)

perm_kv_path = os.path.join(os.path.dirname(__file__), 'ui', 'permission_screen.kv')
if os.path.exists(perm_kv_path):
    Builder.load_file(perm_kv_path)

voice_kv_path = os.path.join(os.path.dirname(__file__), 'ui', 'voice_settings_screen.kv')
if os.path.exists(voice_kv_path):
    Builder.load_file(voice_kv_path)


class BlindNavigationApp(App):
    """
    Main Kivy application for blind navigation assistance.
    Coordinates camera capture, ML inference, path planning, and user feedback.
    """
    
    title = "Blind Navigation Assistant"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_config = get_config()
        self.logger = get_logger("app")
        self.monitor = get_monitor()
        
        # Core components (initialized in on_start)
        self.state_manager = None
        self.camera_manager = None
        self.inference_engine = None
        self.path_planner = None
        self.feedback_controller = None
        self.voice_activation = None
        self.distance_estimator = None
        self.precise_guidance = None
        
        # Processing flag
        self._processing_active = False
        self._update_event = None
    
    def build(self):
        """Build the application UI."""
        self.logger.info("Building application UI")
        
        # Set window properties for accessibility
        Window.clearcolor = (0, 0, 0, 1)  # Black background
        
        # Create screen manager
        self.screen_manager = ScreenManager()
        self.screen_manager.bind(current=self._on_screen_change)
        
        # Add main screen
        self.main_screen = MainScreen(name='main')
        self.main_screen.app = self
        self.screen_manager.add_widget(self.main_screen)
        
        # Add shoe entry choice screen
        self.shoe_entry_choice_screen = ShoeEntryChoiceScreen(name='shoe_entry_choice')
        self.shoe_entry_choice_screen.app = self
        self.screen_manager.add_widget(self.shoe_entry_choice_screen)
        
        # Add shoe type choice screen
        self.shoe_type_choice_screen = ShoeTypeChoiceScreen(name='shoe_type_choice')
        self.shoe_type_choice_screen.app = self
        self.screen_manager.add_widget(self.shoe_type_choice_screen)
        
        # Add shoe setup screen
        self.shoe_setup_screen = ShoeSetupScreen(name='shoe_setup')
        self.shoe_setup_screen.app = self
        self.screen_manager.add_widget(self.shoe_setup_screen)
        
        # Add permission screen
        self.permission_screen = PermissionScreen(name='permission')
        self.permission_screen.app = self
        self.screen_manager.add_widget(self.permission_screen)
        
        # Add voice settings screen
        self.voice_settings_screen = VoiceSettingsScreen(name='voice_settings')
        self.voice_settings_screen.app = self
        self.screen_manager.add_widget(self.voice_settings_screen)
        
        # Set initial screen
        self.screen_manager.current = 'main'
        
        return self.screen_manager
    
    def _on_screen_change(self, instance, value):
        """Called when the active screen changes."""
        self.logger.info(f"Screen changed to: {value}")
        # Global auto-stop removed to allow auto-exiting screens to finish their announcements.
        # Interruption is now handled explicitly by manual user actions.
    
    def on_start(self):
        """Called when the application starts."""
        self.logger.info("Application starting")
        
        try:
            # Initialize core components
            self._init_components()
            
            # Show permission screen if it's the first run
            if self.app_config.is_first_run:
                self.logger.info("First run detected, showing permission screen")
                self.screen_manager.current = 'permission'
                return
            
            # Check if shoe setup is required
            if self.app_config.shoe.require_shoe_analysis:
                self.logger.info("Shoe setup required, showing setup screen")
                self.state_manager.set_state(AppState.SHOE_SETUP)
                self.screen_manager.current = 'shoe_setup'
                self.feedback_controller.speak("shoe_prompt")
            else:
                # Skip to main screen
                self.screen_manager.current = 'main'
                
                # IMPORTANT: Trigger on_enter for the initial screen if it's already set to 'main'
                # or ensure the feedback controller is ready.
                # Since screen_manager.current = 'main' might not trigger on_enter if it was already 'main'
                if self.screen_manager.current == 'main':
                    Clock.schedule_once(lambda dt: self.main_screen.on_enter(), 0.1)
                
                self.feedback_controller.speak("ready")
                
                # Announce last shoe type if it exists
                if self.app_config.last_shoe_type != "无":
                    # Map generic "鞋子" to descriptive "低帮鞋"
                    display_type = self.app_config.last_shoe_type
                    if display_type == "鞋子":
                        display_type = "低帮鞋"
                    last_shoe_msg = f"当前鞋型为：{display_type}"
                    self.feedback_controller.speak_text(last_shoe_msg)
            
            self.logger.info("Application ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self.state_manager.set_state(AppState.ERROR)
    
    def _init_components(self):
        """Initialize all core components."""
        self.logger.info("Initializing components")
        
        # State manager
        self.state_manager = StateManager()
        self.state_manager.set_state(AppState.STARTING)
        
        # Camera manager
        self.camera_manager = CameraManager(
            resolution=self.app_config.camera.resolution,
            target_fps=self.app_config.camera.target_fps,
            max_queue_size=self.app_config.camera.max_queue_size
        )
        
        # Inference engine (handles ML models)
        self.inference_engine = InferenceEngine(
            model_config=self.app_config.model,
            reflection_config=self.app_config.reflection
        )
        
        # Path planner
        self.path_planner = PathPlanner(
            config=self.app_config.path_planning
        )
        
        # Feedback controller (audio + haptic)
        self.feedback_controller = FeedbackController(
            audio_config=self.app_config.audio,
            haptic_config=self.app_config.haptic,
            language=self.app_config.language
        )
        
        # Distance estimator for precise measurements
        self.distance_estimator = DistanceEstimator(
            camera_config=self.app_config.camera
        )
        
        # Precise guidance generator
        self.precise_guidance = PreciseGuidanceGenerator(
            config=self.app_config.distance
        )
        
        # Voice activation (wake word detection)
        if self.app_config.voice.enable_voice_activation:
            try:
                self.voice_activation = VoiceActivation(
                    on_wake=self._on_wake_word,
                    on_command=self._on_voice_command,
                    language=self.app_config.voice.recognition_language
                )
                
                # Check if voice activation is available
                if self.voice_activation.is_available():
                    # Start listening for wake word
                    self.voice_activation.start_listening()
                    self.logger.info("Voice activation enabled")
                else:
                    self.logger.warning("Voice activation not available - microphone or speech recognition unavailable")
                    self.voice_activation = None
            except Exception as e:
                self.logger.error(f"Failed to initialize voice activation: {e}")
                self.voice_activation = None
        
        # Set state to idle
        self.state_manager.set_state(AppState.IDLE)
        
        self.logger.info("All components initialized")
    
    def _on_wake_word(self):
        """Called when wake word is detected."""
        self.logger.info("Wake word detected!")
        
        # Ensure we are on the main screen for navigation and trigger camera logic
        def trigger_camera_logic(dt):
            if hasattr(self, 'main_screen'):
                # Switch to main screen first
                self.screen_manager.current = 'main'
                # Call the button's logic directly to ensure equivalence
                self.main_screen.on_enable_camera()
        
        # Schedule on main thread for UI safety
        Clock.schedule_once(trigger_camera_logic)
    
    def _on_voice_command(self, command: str):
        """Handle voice commands during operation."""
        self.logger.info(f"Voice command: {command}")
        
        if command == "stop":
            self.feedback_controller.speak("voice_command_stop")
            self.stop_navigation()
        elif command == "start":
            self.feedback_controller.speak("voice_command_start")
            self.start_navigation()
        elif command == "reach":
            self.feedback_controller.speak("arrived")
            self.feedback_controller.speak("navigation_completed")
            
            # Stop navigation and update UI safely on main thread
            def handle_arrival(dt):
                self.stop_navigation()
                if hasattr(self, 'main_screen'):
                    self.main_screen._is_navigating = False
                    self.main_screen._set_status("Ready")
                    self.main_screen._update_button_state()
            
            Clock.schedule_once(handle_arrival)
        elif command == "repeat":
            # Repeat last instruction - handled by feedback controller
            pass
    
    def start_navigation(self):
        """Start the navigation assistance."""
        if self._processing_active:
            self.logger.warning("Navigation already active")
            return
        
        self.logger.info("Starting navigation")
        self.state_manager.set_state(AppState.ACTIVE)
        
        # Start camera capture
        self.camera_manager.start()
        
        # Start processing loop
        self._processing_active = True
        # Update FPS to 2 (every 0.5 seconds) for dynamic detection
        fps = 2
        self._update_event = Clock.schedule_interval(
            self._process_frame, 
            1.0 / fps
        )
        
        self.feedback_controller.speak("starting")
    
    def stop_navigation(self):
        """Stop the navigation assistance."""
        if not self._processing_active:
            return
        
        self.logger.info("Stopping navigation")
        
        # Stop processing loop
        self._processing_active = False
        if self._update_event:
            self._update_event.cancel()
            self._update_event = None
        
        # Stop camera
        self.camera_manager.stop()
        
        self.state_manager.set_state(AppState.IDLE)
        self.feedback_controller.speak("paused")
    
    def toggle_navigation(self):
        """Toggle navigation on/off."""
        if self._processing_active:
            self.stop_navigation()
        else:
            self.start_navigation()
    
    def _process_frame(self, dt):
        """
        Process a single frame from the camera.
        Called periodically by Kivy clock.
        """
        if not self._processing_active:
            return
        
        self.monitor.frame_start()
        self.monitor.operation_start("pipeline")
        
        try:
            # Get latest frame from camera
            frame = self.camera_manager.get_frame()
            if frame is None:
                self.monitor.frame_dropped()
                return
            
            # Update camera preview if in debug mode
            if self.app_config.show_camera_preview:
                self.main_screen.update_preview(frame)
            
            # Run inference (water + obstacle detection)
            self.monitor.operation_start("inference")
            detection_result = self.inference_engine.process_frame(frame)
            self.monitor.operation_end("inference")
            
            if detection_result is None:
                return
            
            # Estimate distances to hazards
            self.monitor.operation_start("distance_estimation")
            puddle_infos = self.distance_estimator.estimate_puddle_info(
                detection_result.water_mask,
                frame.shape[:2]
            )
            obstacle_infos = self.distance_estimator.estimate_obstacle_info(
                detection_result.obstacles,
                frame.shape[:2]
            )
            self.monitor.operation_end("distance_estimation")
            
            # Plan safe path
            self.monitor.operation_start("path_planning")
            navigation = self.path_planner.plan(
                water_mask=detection_result.water_mask,
                obstacles=detection_result.obstacles,
                frame_shape=frame.shape
            )
            self.monitor.operation_end("path_planning")
            
            # Generate precise guidance if enabled
            if self.app_config.distance.enable_precise_guidance and (puddle_infos or obstacle_infos):
                precise_instruction = self.precise_guidance.generate_combined_instruction(
                    puddle_infos, obstacle_infos
                )
                # Provide precise feedback
                self.feedback_controller.process_precise_guidance(
                    precise_instruction,
                    self.app_config.language
                )
            else:
                # Provide standard feedback
                self.feedback_controller.process_navigation(navigation)
            
            # Update visualization with RED boxes around obstacles (always enabled)
            self.main_screen.update_debug_overlay(
                detection_result, 
                navigation
            )
            
            # Update state based on safety level
            if navigation.safety_score < self.app_config.path_planning.safety_critical:
                self.state_manager.set_state(AppState.WARNING)
            else:
                self.state_manager.set_state(AppState.ACTIVE)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.monitor.frame_dropped()
        
        finally:
            self.monitor.operation_end("pipeline")
    
    def on_pause(self):
        """Called when app is paused (Android)."""
        self.logger.info("Application paused")
        self.stop_navigation()
        return True  # Allow pause
    
    def on_resume(self):
        """Called when app resumes (Android)."""
        self.logger.info("Application resumed")
        self.feedback_controller.speak("resumed")
    
    def on_stop(self):
        """Called when application is closing."""
        self.logger.info("Application stopping")
        
        # Clean up resources
        self.stop_navigation()
        
        if self.voice_activation:
            self.voice_activation.release()
        
        if self.camera_manager:
            self.camera_manager.release()
        
        if self.inference_engine:
            self.inference_engine.release()
        
        if self.feedback_controller:
            self.feedback_controller.release()
        
        self.logger.info("Application stopped")


def main():
    """Application entry point."""
    # Parse command line arguments
    debug_mode = '--debug' in sys.argv
    set_debug_mode(debug_mode)
    
    # Set up logging
    log_level = 'DEBUG' if debug_mode else 'INFO'
    setup_logger(level=getattr(__import__('logging'), log_level))
    
    # Run the app
    app = BlindNavigationApp()
    app.run()


if __name__ == '__main__':
    main()
