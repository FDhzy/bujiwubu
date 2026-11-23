"""
Main screen UI controller for Blind Navigation Assistant.
"""
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.properties import ObjectProperty, BooleanProperty, StringProperty, ListProperty
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
    # 语音触发开关状态
    _voice_active = BooleanProperty(False)
    _voice_btn_text = StringProperty("语音触发: 关")
    _voice_btn_color = ListProperty([0.5, 0.5, 0.5, 1])
    
    def __init__(self, **kwargs):
        self.logger = get_logger("ui")
        self.config = get_config()
        super().__init__(**kwargs)
        # 预先查找可用中文字体路径
        import os
        self._zh_font_path = None
        for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc']:
            if os.path.exists(fp):
                self._zh_font_path = fp
                break
    
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
                "欢迎来到主界面。屏幕左上角是语音触发开关，"
                "右上角是声音设置。屏幕中央分布着两个大按钮："
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
    
    def on_toggle_voice_activation(self):
        """Handle voice activation toggle button press."""
        if self.app is None:
            return
        
        if self._voice_active:
            # 关闭语音触发
            self._voice_active = False
            self._voice_btn_text = "语音触发: 关"
            self._voice_btn_color = [0.5, 0.5, 0.5, 1]  # 灰色
            # 停止语音监听
            if self.app.voice_activation:
                self.app.voice_activation.stop_listening()
            if self.app.feedback_controller:
                self.app.feedback_controller.speak_text("语音触发已关闭")
            self.logger.info("Voice activation disabled by user")
        else:
            # 开启语音触发
            self._voice_active = True
            self._voice_btn_text = "语音触发: 开"
            self._voice_btn_color = [0.2, 0.7, 0.3, 1]  # 绿色
            # 启动语音监听
            if self.app.voice_activation and self.app.voice_activation.is_available():
                self.app.voice_activation.start_listening()
                if self.app.feedback_controller:
                    self.app.feedback_controller.speak_text("语音触发已开启，请说唤醒词")
            else:
                # 尝试初始化语音激活
                if self.app.feedback_controller:
                    self.app.feedback_controller.speak_text("正在初始化语音触发")
                Clock.schedule_once(lambda dt: self._try_init_voice(), 0.3)
            self.logger.info("Voice activation enabled by user")
    
    def _try_init_voice(self):
        """尝试初始化并启动语音激活。"""
        if self.app is None:
            return
        try:
            from core.voice_activation import VoiceActivation
            self.app.voice_activation = VoiceActivation(
                on_wake=self.app._on_wake_word,
                on_command=self.app._on_voice_command,
                language=self.app.app_config.voice.recognition_language
            )
            if self.app.voice_activation.is_available():
                self.app.voice_activation.start_listening()
                if self.app.feedback_controller:
                    self.app.feedback_controller.speak_text("语音触发已就绪，请说唤醒词")
            else:
                self._voice_active = False
                self._voice_btn_text = "语音触发: 关"
                self._voice_btn_color = [0.5, 0.5, 0.5, 1]
                if self.app.feedback_controller:
                    self.app.feedback_controller.speak_text("语音触发不可用，请检查麦克风")
                self.logger.warning("Voice activation unavailable - microphone not found")
        except Exception as e:
            self._voice_active = False
            self._voice_btn_text = "语音触发: 关"
            self._voice_btn_color = [0.5, 0.5, 0.5, 1]
            self.logger.error(f"Failed to init voice activation: {e}")
    
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
    
    # 中文类别名称映射
    CATEGORY_ZH = {
        'person':       '行人',
        'bicycle':      '自行车',
        'car':          '汽车',
        'motorcycle':   '摩托车',
        'stairs':       '台阶',
        'wire':         '电线',
        'bench':        '长椅',
        'speed_bump':   '减速带',
        'pothole':      '坑洞',
        'utility_pole': '电线杆',
        'tree':         '树木',
        'obstacle':     '障碍物',
    }

    # 英文显示名（红框标签用，避免字体问题）
    CATEGORY_EN_DISPLAY = {
        'person':       'person(行人)',
        'bicycle':      'bicycle(自行车)',
        'car':          'car(汽车)',
        'motorcycle':   'moto(摩托车)',
        'stairs':       'stairs(台阶)',
        'wire':         'wire(电线)',
        'bench':        'bench(长椅)',
        'speed_bump':   'bump(减速带)',
        'pothole':      'pothole(坑洞)',
        'utility_pole': 'pole(电线杆)',
        'tree':         'tree(树木)',
        'obstacle':     'obstacle(障碍物)',
    }

    # 中下区域起始比例（帧高度的此比例以下视为"中下部分"）
    # 0.4 表示从40%处往下（即下方60%区域）才绘框/播报
    LOWER_REGION_START = 0.4
    # 检测框与中下区域交叠面积占检测框总面积的最低比例
    LOWER_OVERLAP_RATIO = 0.15

    def _is_in_lower_region(self, x, y, w, h, frame_w, frame_h):
        """
        判断检测框是否有足够面积落在画面中下部分。
        中下部分定义为 y > frame_h * LOWER_REGION_START 的区域。
        交叠面积占检测框面积之比须 >= LOWER_OVERLAP_RATIO 才返回 True。
        """
        lower_y_start = frame_h * self.LOWER_REGION_START
        # 检测框与中下区域在 y 轴的交叠
        overlap_y_start = max(y, lower_y_start)
        overlap_y_end = y + h
        if overlap_y_end <= overlap_y_start:
            return False
        overlap_h = overlap_y_end - overlap_y_start
        overlap_area = overlap_h * w
        box_area = w * h
        if box_area == 0:
            return False
        return (overlap_area / box_area) >= self.LOWER_OVERLAP_RATIO

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
                    self.logger.debug(f"[UI] obstacle class_name={getattr(obstacle,'class_name','N/A')!r} class_id={getattr(obstacle,'class_id','N/A')}")
                    x, y, w, h = obstacle.bbox

                    # ── 只绘制落在画面中下部分的障碍物 ──
                    if not self._is_in_lower_region(
                        x, y, w, h,
                        detection_result.frame_width,
                        detection_result.frame_height
                    ):
                        continue
                    
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
                        
                        cn = obstacle.class_name or 'obstacle'
                        display_name = self.CATEGORY_EN_DISPLAY.get(cn, cn)
                        label_text = f"{display_name} {obstacle.confidence:.0%}"
                        
                        if self._zh_font_path:
                            label = CoreLabel(text=label_text, font_size=16, bold=True, font_name=self._zh_font_path)
                        else:
                            label = CoreLabel(text=label_text, font_size=16, bold=True)
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

                        # ── 只绘制落在画面中下部分的水坑 ──
                        if not self._is_in_lower_region(
                            x, y, w, h,
                            detection_result.frame_width,
                            detection_result.frame_height
                        ):
                            continue
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

            # Draw traffic light bounding box if detected
            if detection_result and detection_result.traffic_light is not None:
                tl = detection_result.traffic_light
                from models.traffic_light_detector import TrafficLightColor
                if tl.color != TrafficLightColor.UNKNOWN and tl.bbox is not None:
                    tx, ty, tw, th = tl.bbox
                    sx = debug_overlay.width  / detection_result.frame_width
                    sy = debug_overlay.height / detection_result.frame_height
                    bx = debug_overlay.x + tx * sx
                    by = debug_overlay.y + (detection_result.frame_height - ty - th) * sy
                    bw, bh = tw * sx, th * sy
                    # Colour matches traffic light colour
                    if tl.color == TrafficLightColor.GREEN:
                        Color(0.1, 0.95, 0.2, 1)
                    elif tl.color == TrafficLightColor.RED:
                        Color(1, 0.1, 0.1, 1)
                    else:  # yellow
                        Color(1, 0.85, 0, 1)
                    Line(rectangle=(bx, by, bw, bh), width=3)
                    # Label
                    from kivy.graphics import Rectangle
                    from kivy.core.text import Label as CoreLabel
                    tl_label_map = {'red': 'RED(红灯)', 'yellow': 'YELLOW(黄灯)', 'green': 'GREEN(绿灯)'}
                    lbl_text = tl_label_map.get(tl.color.value, tl.color.value)
                    lbl = CoreLabel(text=lbl_text, font_size=15, bold=True)
                    lbl.refresh()
                    Color(0, 0, 0, 0.7)
                    Rectangle(pos=(bx, by - lbl.texture.height - 6),
                              size=(lbl.texture.width + 8, lbl.texture.height + 4))
                    Color(1, 1, 1, 1)
                    Rectangle(texture=lbl.texture,
                              pos=(bx + 4, by - lbl.texture.height - 4),
                              size=lbl.texture.size)
        
        # 更新顶部状态栏：只显示中下区域内检测到的障碍物类别
        if detection_result and detection_result.obstacles:
            names = []
            seen = set()
            for obs in detection_result.obstacles:
                ox, oy, ow, oh = obs.bbox
                if not self._is_in_lower_region(
                    ox, oy, ow, oh,
                    detection_result.frame_width,
                    detection_result.frame_height
                ):
                    continue
                cn = getattr(obs, 'class_name', '') or 'obstacle'
                zh = self.CATEGORY_ZH.get(cn, cn)
                if zh not in seen:
                    seen.add(zh)
                    names.append(zh)
            if names:
                status_text = '检测到：' + '、'.join(names)
                self._set_status(status_text)
            else:
                self._set_status('避障中')
        else:
            self._set_status('避障中')
    
    def hide_debug_overlay(self):
        """Hide the debug overlay."""
        debug_overlay = self.ids.get('debug_overlay')
        if debug_overlay:
            debug_overlay.opacity = 0
            debug_overlay.canvas.clear()

    # ── Traffic light UI ──────────────────────────────────────────────────────

    # Colour used in the status bar for each traffic light state
    _TL_COLOR_TEXT = {
        'red':    '🔴 红灯',
        'yellow': '🟡 黄灯',
        'green':  '🟢 绿灯',
    }
    _TL_KIVY_COLOR = {
        'red':    (1, 0.15, 0.15, 1),
        'yellow': (1, 0.85, 0, 1),
        'green':  (0.1, 0.9, 0.2, 1),
    }

    def update_traffic_light_status(self, traffic_light_result):
        """
        Update the top status bar to show detected traffic light colour.
        Called from main.py after each frame.

        Args:
            traffic_light_result: TrafficLightResult or None
        """
        if traffic_light_result is None:
            return  # Keep whatever obstacle status is already shown

        try:
            from models.traffic_light_detector import TrafficLightColor
            color = traffic_light_result.color
            if color == TrafficLightColor.UNKNOWN:
                return

            color_key = color.value  # 'red' / 'yellow' / 'green'
            label_text = self._TL_COLOR_TEXT.get(color_key, color_key)
            self._set_status(label_text)

            # Optionally tint the status bar background
            status_label = self.ids.get('status_label')
            if status_label is not None:
                rgb = self._TL_KIVY_COLOR.get(color_key, (1, 1, 1, 1))
                try:
                    status_label.color = rgb
                except Exception:
                    pass  # widget may not support .color property

        except Exception as e:
            self.logger.debug(f"update_traffic_light_status error: {e}")
