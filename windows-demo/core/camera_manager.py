"""
Camera manager for capturing frames from device camera.
Supports both desktop (OpenCV) and Android (Kivy/pyjnius) cameras.
"""
import threading
from queue import Queue, Empty, Full
from typing import Optional, Tuple
import numpy as np
import time

from utils.logger import get_logger
from utils.performance_monitor import get_monitor

# Try to import OpenCV for desktop support
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Check if running on Android
try:
    from kivy.utils import platform
    if platform == 'android':
        from android.permissions import request_permissions, Permission
        from jnius import autoclass
        ANDROID = True
    else:
        ANDROID = False
except (ImportError, Exception):
    ANDROID = False


class CameraManager:
    """
    Manages camera capture with a frame queue for processing.
    Supports both desktop (OpenCV) and Android platforms.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        target_fps: int = 15,
        max_queue_size: int = 3,
        camera_index: int = 0
    ):
        """
        Initialize camera manager.
        
        Args:
            resolution: (width, height) for camera capture
            target_fps: Target frames per second
            max_queue_size: Maximum frames to buffer
            camera_index: Camera device index (0 = default)
        """
        self.logger = get_logger("camera")
        self.monitor = get_monitor()
        
        self.resolution = resolution
        self.target_fps = target_fps
        self.max_queue_size = max_queue_size
        self.camera_index = camera_index
        
        # Frame queue for processing
        self._frame_queue: Queue = Queue(maxsize=max_queue_size)
        
        # Camera state
        self._capture = None
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Frame timing
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0
        
        self.logger.info(
            f"CameraManager initialized: {resolution[0]}x{resolution[1]} @ {target_fps}fps"
        )
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._is_running:
                self.logger.warning("Camera already running")
                return True
            
            try:
                if ANDROID:
                    success = self._start_android_camera()
                elif CV2_AVAILABLE:
                    success = self._start_opencv_camera()
                else:
                    self.logger.error("No camera backend available")
                    return False
                
                if success:
                    self._is_running = True
                    self._start_capture_thread()
                    self.logger.info("Camera started successfully")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Failed to start camera: {e}")
                return False
    
    def _start_opencv_camera(self) -> bool:
        """Initialize OpenCV camera capture."""
        self._capture = cv2.VideoCapture(self.camera_index)
        
        if not self._capture.isOpened():
            self.logger.error("Failed to open OpenCV camera")
            return False
        
        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Verify settings
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"OpenCV camera: {actual_width}x{actual_height}")
        
        return True
    
    def _start_android_camera(self) -> bool:
        """Initialize Android camera via pyjnius."""
        # Request camera permission
        request_permissions([Permission.CAMERA])
        
        # Android camera implementation
        # Uses Kivy's native camera or direct Android API
        try:
            from kivy.uix.camera import Camera as KivyCamera
            # Kivy camera will be managed by the UI
            self.logger.info("Android camera initialized via Kivy")
            return True
        except Exception as e:
            self.logger.error(f"Android camera init failed: {e}")
            return False
    
    def _start_capture_thread(self):
        """Start background thread for continuous capture."""
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._capture_thread.start()
    
    def _capture_loop(self):
        """Continuous capture loop running in background thread."""
        self.logger.info("Capture thread started")
        
        while self._is_running:
            try:
                # Throttle capture to target FPS
                current_time = time.perf_counter()
                elapsed = current_time - self._last_frame_time
                
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                
                self._last_frame_time = time.perf_counter()
                
                # Capture frame
                frame = self._capture_frame()
                
                if frame is not None:
                    # Try to add to queue, drop if full
                    try:
                        self._frame_queue.put_nowait(frame)
                    except Full:
                        # Drop oldest frame and add new one
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                            self.monitor.frame_dropped()
                        except Empty:
                            pass
                
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors
        
        self.logger.info("Capture thread stopped")
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            BGR image as numpy array, or None if capture failed
        """
        if not CV2_AVAILABLE or self._capture is None:
            return None
        
        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            return None
        
        return frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the queue.
        Non-blocking - returns None if no frame available.
        
        Returns:
            BGR image as numpy array, or None
        """
        try:
            # Get latest frame, discarding older ones
            frame = None
            while True:
                try:
                    frame = self._frame_queue.get_nowait()
                except Empty:
                    break
            return frame
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get a frame from the queue, waiting if necessary.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            BGR image as numpy array, or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stop(self):
        """Stop camera capture."""
        self.logger.info("Stopping camera")
        
        with self._lock:
            self._is_running = False
        
        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
    
    def release(self):
        """Release camera resources."""
        self.stop()
        
        if CV2_AVAILABLE and self._capture is not None:
            self._capture.release()
            self._capture = None
        
        self.logger.info("Camera released")
    
    @property
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._is_running
    
    @property
    def queue_size(self) -> int:
        """Get current number of frames in queue."""
        return self._frame_queue.qsize()
