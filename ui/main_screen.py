"""
Main screen UI controller for Blind Navigation Assistant.
"""
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.properties import ObjectProperty, BooleanProperty
import numpy as np

from utils.config import get_config, AppState
from utils.logger import get_logger


class MainScreen(Screen):
    """
    Main application screen with camera preview and controls.
    Designed for accessibility with large buttons and high contrast.
    """
    
    app = ObjectProperty(None)
    _is_navigating = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        self.logger = get_logger("ui")
        self.config = get_config()
        super().__init__(**kwargs)
    
    def on_kv_post(self, base_widget):
        """Called after KV file is loaded."""
        self.logger.info("MainScreen initialized")
        self._update_button_state()
    
    def on_enter(self):
        """Called when screen is displayed."""
        self.logger.info("Entering main screen")
        
        # Announce instructions via TTS
        if self.app and self.app.feedback_controller:
            instruction_text = (
                "欢迎来到主界面。屏幕中心分布着两个大按钮："
                "上方是录入鞋子，下方是开启相机。"
            )
            self.app.feedback_controller.speak_text(instruction_text)
    
    def on_register_shoes(self):
        """Handle 'Register Shoes' button press."""
        if self.app is None:
            return
        
        # Stop previous speech before navigating
        if self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在进入鞋子录入界面")
            
        # Switch to shoe entry choice screen
        self.app.state_manager.set_state(AppState.SHOE_SETUP)
        self.app.screen_manager.current = 'shoe_entry_choice'
    
    def on_enable_camera(self):
        """Handle 'Enable Camera' button press."""
        if self.app is None:
            return
            
        # Stop any ongoing speech before new navigation prompt
        if self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("已开启智能避障，请将摄像头对准路面，愿君听雨辨途，不误涉云端碎影")
            
        # Start navigation/intelligent obstacle avoidance
        self._is_navigating = True
        self.app.start_navigation()
        self._set_status("避障中")
        self._update_button_state()

    def on_exit_navigation_click(self):
        """Handle 'Exit' button press during navigation."""
        self.logger.info("Navigation exit button pressed - stopping all functions immediately")
        if self.app is None:
            return
        
        # IMMEDIATELY stop audio guidance (clear queue and stop speaking)
        if self.app.feedback_controller:
            self.logger.info("Stopping audio guidance immediately...")
            self.app.feedback_controller.audio.stop()  # Stop audio thread and clear queue
        
        # IMMEDIATELY stop navigation
        if self._is_navigating:
            self.logger.info("Stopping navigation immediately...")
            self.app.stop_navigation()
        
        # Reset navigation state
        self._is_navigating = False
        
        # Update UI
        self._set_status("已停止")
        self._update_button_state()
        
        # Schedule application shutdown immediately (no voice announcement)
        self.logger.info("Scheduling application shutdown...")
        Clock.schedule_once(lambda dt: self._shutdown_application(), 0.3)

    def _shutdown_application(self):
        """Actually stop the Kivy application."""
        if self.app:
            self.app.stop()

    def on_start_stop(self):
        """Handle start/stop button press."""
        if self.app is None:
            return
        
        # Stop any ongoing speech before new state announcement
        if self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
        
        self._is_navigating = not self._is_navigating
        
        if self._is_navigating:
            self.app.start_navigation()
            if self.app.feedback_controller:
                self.app.feedback_controller.speak_text("避障已启动")
            self._set_status("避障中")
        else:
            self.app.stop_navigation()
            if self.app.feedback_controller:
                self.app.feedback_controller.speak_text("避障已停止")
            self._set_status("Ready")
        
        self._update_button_state()
    
    def on_emergency(self):
        """Handle emergency stop button press."""
        if self.app is None:
            return
        
        self.logger.warning("Emergency stop triggered")
        
        # IMMEDIATELY stop all audio
        if self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("紧急停止")
        
        self._is_navigating = False
        self.app.stop_navigation()
        self._set_status("STOPPED")
        self._update_button_state()
        
        # Trigger emergency feedback
        if self.app.feedback_controller:
            self.app.feedback_controller.emergency_stop()
    
    def on_settings(self):
        """Handle settings button press."""
        self.logger.info("Settings requested")
        # TODO: Implement settings screen
        pass
    
    def on_voice_settings(self):
        """Handle voice settings button press."""
        self.logger.info("Voice settings requested")
        if self.app is None:
            return
        
        # Stop previous speech before navigating
        if self.app.feedback_controller:
            self.app.feedback_controller.stop_audio()
            self.app.feedback_controller.speak_text("正在进入声音设置界面")
        
        # Switch to voice settings screen
        self.app.screen_manager.current = 'voice_settings'
    
    def _update_button_state(self):
        """Update button appearance based on navigation state."""
        start_btn = self.ids.get('start_stop_btn')
        if start_btn:
            if self._is_navigating:
                start_btn.text = "STOP"
                start_btn.background_color = (0.8, 0.4, 0.1, 1)  # Orange
            else:
                start_btn.text = "START"
                start_btn.background_color = (0.2, 0.6, 0.2, 1)  # Green
    
    def _set_status(self, text: str):
        """Update status label text."""
        status_label = self.ids.get('status_label')
        if status_label:
            status_label.text = text
    
    def update_status(self, state: AppState):
        """Update status display based on app state."""
        status_map = {
            AppState.IDLE: ("Ready", (1, 1, 1, 1)),
            AppState.STARTING: ("Starting...", (1, 1, 0.5, 1)),
            AppState.ACTIVE: ("Active", (0.5, 1, 0.5, 1)),
            AppState.PROCESSING: ("Processing...", (0.5, 0.5, 1, 1)),
            AppState.WARNING: ("WARNING", (1, 0.5, 0.2, 1)),
            AppState.PAUSED: ("Paused", (0.7, 0.7, 0.7, 1)),
            AppState.ERROR: ("ERROR", (1, 0.3, 0.3, 1)),
        }
        
        text, color = status_map.get(state, ("Unknown", (1, 1, 1, 1)))
        
        status_label = self.ids.get('status_label')
        if status_label:
            status_label.text = text
            status_label.color = color
    
    def update_preview(self, frame: np.ndarray):
        """
        Update the camera preview with a new frame.
        
        Args:
            frame: BGR image from camera (numpy array)
        """
        if frame is None:
            return
        
        camera_preview = self.ids.get('camera_preview')
        if camera_preview is None:
            return
        
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            else:
                frame_rgb = frame
            
            # Flip vertically for Kivy texture
            frame_flipped = np.flipud(frame_rgb)
            
            # Create texture
            height, width = frame_flipped.shape[:2]
            texture = Texture.create(size=(width, height), colorfmt='rgb')
            texture.blit_buffer(
                frame_flipped.tobytes(),
                colorfmt='rgb',
                bufferfmt='ubyte'
            )
            
            # Update preview
            camera_preview.texture = texture
            
        except Exception as e:
            self.logger.error(f"Failed to update preview: {e}")
    
    def update_debug_overlay(self, detection_result, navigation):
        """
        Update visualization overlay with red boxes around detected obstacles.
        Always shows obstacles when camera is active.
        
        Args:
            detection_result: Detection results from inference engine
            navigation: Navigation plan from path planner
        """
        debug_overlay = self.ids.get('debug_overlay')
        if debug_overlay is None:
            return
        
        # Make overlay visible
        debug_overlay.opacity = 1
        
        # Clear previous drawings
        debug_overlay.canvas.clear()
        
        with debug_overlay.canvas:
            # Always draw RED boxes around detected obstacles
            if detection_result and detection_result.obstacles:
                for obstacle in detection_result.obstacles:
                    x, y, w, h = obstacle.bbox
                    
                    # Scale to widget size
                    sx = debug_overlay.width / detection_result.frame_width
                    sy = debug_overlay.height / detection_result.frame_height
                    
                    # Calculate box coordinates
                    box_x = debug_overlay.x + x * sx
                    box_y = debug_overlay.y + (detection_result.frame_height - y - h) * sy
                    box_w = w * sx
                    box_h = h * sy
                    
                    # Draw RED border around obstacle (pure red, fully opaque)
                    Color(1, 0, 0, 1)  # Pure red RGB(255, 0, 0)
                    Line(
                        rectangle=(box_x, box_y, box_w, box_h),
                        width=3  # Thick line for visibility
                    )
                    
                    # Draw corner markers for emphasis
                    corner_size = min(box_w, box_h) * 0.15
                    # Top-left corner
                    Line(points=[box_x, box_y + box_h, box_x + corner_size, box_y + box_h], width=4)
                    Line(points=[box_x, box_y + box_h, box_x, box_y + box_h - corner_size], width=4)
                    # Top-right corner
                    Line(points=[box_x + box_w - corner_size, box_y + box_h, box_x + box_w, box_y + box_h], width=4)
                    Line(points=[box_x + box_w, box_y + box_h, box_x + box_w, box_y + box_h - corner_size], width=4)
                    # Bottom-left corner
                    Line(points=[box_x, box_y, box_x + corner_size, box_y], width=4)
                    Line(points=[box_x, box_y, box_x, box_y + corner_size], width=4)
                    # Bottom-right corner
                    Line(points=[box_x + box_w - corner_size, box_y, box_x + box_w, box_y], width=4)
                    Line(points=[box_x + box_w, box_y, box_x + box_w, box_y + corner_size], width=4)
                    
                    # Draw label with red background (optional, if class_name available)
                    if hasattr(obstacle, 'class_name') and obstacle.class_name:
                        from kivy.graphics import Rectangle
                        from kivy.core.text import Label as CoreLabel
                        
                        label_text = f"{obstacle.class_name} {obstacle.confidence:.0%}"
                        label = CoreLabel(text=label_text, font_size=14, bold=True)
                        label.refresh()
                        
                        # Label background
                        Color(1, 0, 0, 0.8)  # Red background
                        Rectangle(
                            pos=(box_x, box_y + box_h + 2),
                            size=(label.texture.width + 8, label.texture.height + 4)
                        )
                        
                        # Label text
                        Color(1, 1, 1, 1)  # White text
                        Rectangle(
                            texture=label.texture,
                            pos=(box_x + 4, box_y + box_h + 4),
                            size=label.texture.size
                        )
            
            # Draw BLUE boxes around detected water puddles
            if detection_result and hasattr(detection_result, 'water_mask') and detection_result.water_mask is not None:
                import cv2 as _cv2
                import numpy as _np
                water_mask = detection_result.water_mask
                if _np.any(water_mask > 0):
                    sx = debug_overlay.width / detection_result.frame_width
                    sy = debug_overlay.height / detection_result.frame_height
                    contours, _ = _cv2.findContours(
                        water_mask.astype(_np.uint8),
                        _cv2.RETR_EXTERNAL,
                        _cv2.CHAIN_APPROX_SIMPLE
                    )
                    min_area = detection_result.frame_width * detection_result.frame_height * 0.01
                    for cnt in contours:
                        if _cv2.contourArea(cnt) < min_area:
                            continue
                        x, y, w, h = _cv2.boundingRect(cnt)
                        box_x = debug_overlay.x + x * sx
                        box_y = debug_overlay.y + (detection_result.frame_height - y - h) * sy
                        box_w = w * sx
                        box_h = h * sy
                        # Draw BLUE border around puddle
                        Color(0, 0.5, 1, 1)  # Blue
                        Line(rectangle=(box_x, box_y, box_w, box_h), width=2)
                        # Draw label
                        from kivy.graphics import Rectangle
                        from kivy.core.text import Label as CoreLabel
                        label = CoreLabel(text="water", font_size=13, bold=True)
                        label.refresh()
                        Color(0, 0.5, 1, 0.8)
                        Rectangle(
                            pos=(box_x, box_y - label.texture.height - 6),
                            size=(label.texture.width + 8, label.texture.height + 4)
                        )
                        Color(1, 1, 1, 1)
                        Rectangle(
                            texture=label.texture,
                            pos=(box_x + 4, box_y - label.texture.height - 4),
                            size=label.texture.size
                        )

            # Draw recommended direction arrow (only in debug mode)
            if self.config.debug_mode and navigation:
                Color(0.3, 1, 0.3, 0.8)  # Green
                center_x = debug_overlay.center_x
                center_y = debug_overlay.y + debug_overlay.height * 0.3
                
                # Arrow pointing in recommended direction
                import math
                angle_rad = math.radians(90 - navigation.turn_angle)  # 90 = straight up
                arrow_length = debug_overlay.height * 0.2
                
                end_x = center_x + arrow_length * math.cos(angle_rad)
                end_y = center_y + arrow_length * math.sin(angle_rad)
                
                Line(points=[center_x, center_y, end_x, end_y], width=3)
    
    def hide_debug_overlay(self):
        """Hide the debug overlay."""
        debug_overlay = self.ids.get('debug_overlay')
        if debug_overlay:
            debug_overlay.opacity = 0
            debug_overlay.canvas.clear()
