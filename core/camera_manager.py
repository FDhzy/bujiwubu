"""
Camera manager for capturing frames from device camera.
Supports:
  - Local webcam (OpenCV VideoCapture index)
  - IP camera over RTSP or HTTP MJPEG (OpenCV VideoCapture URL)
  - Android camera (Kivy / pyjnius)
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

    Source priority
    ---------------
    1. ip_camera_url != ""  →  OpenCV VideoCapture(url)  (RTSP / HTTP MJPEG)
    2. ANDROID              →  Android camera via Kivy
    3. Otherwise            →  OpenCV VideoCapture(camera_index)

    IP camera auto-reconnect
    ------------------------
    When the stream is lost the capture thread waits
    `reconnect_delay` seconds and retries automatically.
    Set max_retries=0 for unlimited retries.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        target_fps: int = 15,
        max_queue_size: int = 3,
        camera_index: int = 0,
        ip_camera_url: str = "",
        reconnect_delay: float = 3.0,
        max_retries: int = 0,
    ):
        """
        Args:
            resolution        : (width, height) for camera capture
            target_fps        : Target frames per second
            max_queue_size    : Maximum frames to buffer
            camera_index      : Camera device index (used when ip_camera_url is empty)
            ip_camera_url     : RTSP / HTTP URL of IP camera.  Empty = local cam.
            reconnect_delay   : Seconds between reconnect attempts
            max_retries       : 0 = unlimited reconnect attempts
        """
        self.logger = get_logger("camera")
        self.monitor = get_monitor()

        self.resolution       = resolution
        self.target_fps       = target_fps
        self.max_queue_size   = max_queue_size
        self.camera_index     = camera_index
        self.ip_camera_url    = ip_camera_url.strip()
        self.reconnect_delay  = reconnect_delay
        self.max_retries      = max_retries

        # Frame queue for processing
        self._frame_queue: Queue = Queue(maxsize=max_queue_size)

        # Camera state
        self._capture              = None
        self._is_running           = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock                 = threading.Lock()

        # Frame timing
        self._frame_interval  = 1.0 / target_fps
        self._last_frame_time = 0.0

        # Reconnect state (IP camera only)
        self._retry_count     = 0
        self._stream_ok       = False

        src = f"IP={self.ip_camera_url}" if self.ip_camera_url else f"index={camera_index}"
        self.logger.info(
            f"CameraManager init: {resolution[0]}x{resolution[1]} "
            f"@ {target_fps}fps  source={src}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start camera capture."""
        with self._lock:
            if self._is_running:
                self.logger.warning("Camera already running")
                return True

            try:
                if self.ip_camera_url:
                    success = self._start_ip_camera()
                elif ANDROID:
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

    def stop(self):
        """Stop camera capture."""
        self.logger.info("Stopping camera")
        with self._lock:
            self._is_running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)

        self._drain_queue()

    def release(self):
        """Release all camera resources."""
        self.stop()
        self._release_capture()
        self.logger.info("Camera released")

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking: return the newest queued frame (discard stale ones)."""
        try:
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
        """Blocking: wait up to *timeout* seconds for a frame."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def queue_size(self) -> int:
        return self._frame_queue.qsize()

    @property
    def source_description(self) -> str:
        """Human-readable description of the current video source."""
        if self.ip_camera_url:
            return f"IP Camera: {self.ip_camera_url}"
        return f"Local Camera (index={self.camera_index})"

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _start_opencv_camera(self) -> bool:
        """Open a local webcam via OpenCV."""
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self.logger.error(f"Cannot open local camera index={self.camera_index}")
            return False
        self._apply_capture_settings()
        self._stream_ok = True
        return True

    def _start_ip_camera(self) -> bool:
        """
        Open an IP camera stream via OpenCV.
        Supports RTSP, HTTP MJPEG, and single-frame HTTP endpoints.
        Returns True even if the first connect fails – the capture thread
        will keep retrying automatically.
        """
        if not CV2_AVAILABLE:
            self.logger.error("OpenCV not available – cannot use IP camera")
            return False

        self.logger.info(f"Connecting to IP camera: {self.ip_camera_url}")
        connected = self._open_ip_stream()
        if connected:
            self.logger.info("IP camera connected")
        else:
            self.logger.warning(
                "IP camera not reachable yet – will retry in capture thread"
            )
        # Return True so the capture thread starts regardless; it will retry.
        return True

    def _open_ip_stream(self) -> bool:
        """Try to open the IP camera VideoCapture.  Returns True on success."""
        try:
            self._release_capture()
            cap = cv2.VideoCapture(self.ip_camera_url)
            # Give OpenCV a moment to establish the connection
            time.sleep(0.5)
            if not cap.isOpened():
                cap.release()
                return False
            # Quick test-read
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return False
            self._capture = cap
            self._apply_capture_settings()
            self._stream_ok = True
            self._retry_count = 0
            return True
        except Exception as e:
            self.logger.error(f"IP camera open error: {e}")
            return False

    def _start_android_camera(self) -> bool:
        """Initialize Android camera via Kivy / pyjnius."""
        request_permissions([Permission.CAMERA])
        try:
            from kivy.uix.camera import Camera as KivyCamera  # noqa: F401
            self.logger.info("Android camera initialised via Kivy")
            return True
        except Exception as e:
            self.logger.error(f"Android camera init failed: {e}")
            return False

    def _apply_capture_settings(self):
        """Apply resolution / FPS hints to an open VideoCapture."""
        if self._capture is None:
            return
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._capture.set(cv2.CAP_PROP_FPS,          self.target_fps)
        # Reduce internal buffer to minimise latency
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Capture settings applied: {w}x{h}")

    # ──────────────────────────────────────────────────────────────────────
    # Capture thread
    # ──────────────────────────────────────────────────────────────────────

    def _start_capture_thread(self):
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture",
        )
        self._capture_thread.start()

    def _capture_loop(self):
        """Background thread: grab frames and push them into the queue."""
        self.logger.info("Capture thread started")

        while self._is_running:
            try:
                # ── reconnect if stream was lost ──────────────────────────
                if self.ip_camera_url and not self._stream_ok:
                    if self.max_retries and self._retry_count >= self.max_retries:
                        self.logger.error(
                            f"IP camera: max retries ({self.max_retries}) reached, giving up"
                        )
                        self._is_running = False
                        break

                    self._retry_count += 1
                    self.logger.info(
                        f"IP camera reconnect attempt #{self._retry_count} "
                        f"(waiting {self.reconnect_delay}s)"
                    )
                    time.sleep(self.reconnect_delay)

                    if self._open_ip_stream():
                        self.logger.info("IP camera reconnected")
                    else:
                        continue  # keep retrying

                # ── throttle to target FPS ────────────────────────────────
                now = time.perf_counter()
                elapsed = now - self._last_frame_time
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                self._last_frame_time = time.perf_counter()

                # ── capture frame ─────────────────────────────────────────
                frame = self._capture_frame()

                if frame is not None:
                    try:
                        self._frame_queue.put_nowait(frame)
                    except Full:
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                            self.monitor.frame_dropped()
                        except Empty:
                            pass
                else:
                    # Frame grab failed → mark stream as broken for IP cam
                    if self.ip_camera_url:
                        self.logger.warning("IP camera frame lost, will reconnect")
                        self._stream_ok = False
                        self._release_capture()

            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                if self.ip_camera_url:
                    self._stream_ok = False
                    self._release_capture()
                time.sleep(0.1)

        self.logger.info("Capture thread stopped")

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Read one frame from the current VideoCapture."""
        if not CV2_AVAILABLE or self._capture is None:
            return None
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return None
        return frame

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _release_capture(self):
        if CV2_AVAILABLE and self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._stream_ok = False

    def _drain_queue(self):
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
