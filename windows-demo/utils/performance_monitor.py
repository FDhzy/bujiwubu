"""
Performance monitoring utilities.
Tracks FPS, latency, and resource usage.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
from threading import Lock


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    fps: float = 0.0
    inference_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    frame_drops: int = 0
    memory_mb: float = 0.0


class PerformanceMonitor:
    """
    Monitors application performance metrics.
    Thread-safe implementation for multi-threaded apps.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to average over
        """
        self._window_size = window_size
        self._lock = Lock()
        
        # Frame timing
        self._frame_times: Deque[float] = deque(maxlen=window_size)
        self._last_frame_time: Optional[float] = None
        
        # Latency tracking by operation name
        self._latencies: Dict[str, Deque[float]] = {}
        self._operation_starts: Dict[str, float] = {}
        
        # Counters
        self._frame_drops = 0
        self._total_frames = 0
    
    def frame_start(self) -> None:
        """Mark the start of a new frame."""
        current_time = time.perf_counter()
        
        with self._lock:
            if self._last_frame_time is not None:
                frame_duration = current_time - self._last_frame_time
                self._frame_times.append(frame_duration)
            
            self._last_frame_time = current_time
            self._total_frames += 1
    
    def frame_dropped(self) -> None:
        """Record a dropped frame."""
        with self._lock:
            self._frame_drops += 1
    
    def operation_start(self, name: str) -> None:
        """
        Mark the start of a timed operation.
        
        Args:
            name: Operation identifier (e.g., 'water_detection', 'path_planning')
        """
        with self._lock:
            self._operation_starts[name] = time.perf_counter()
    
    def operation_end(self, name: str) -> float:
        """
        Mark the end of a timed operation and return latency in ms.
        
        Args:
            name: Operation identifier
            
        Returns:
            Latency in milliseconds
        """
        end_time = time.perf_counter()
        
        with self._lock:
            if name not in self._operation_starts:
                return 0.0
            
            latency = (end_time - self._operation_starts[name]) * 1000  # to ms
            
            if name not in self._latencies:
                self._latencies[name] = deque(maxlen=self._window_size)
            
            self._latencies[name].append(latency)
            del self._operation_starts[name]
            
            return latency
    
    def get_fps(self) -> float:
        """Get current frames per second."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            if avg_frame_time <= 0:
                return 0.0
            
            return 1.0 / avg_frame_time
    
    def get_latency(self, name: str) -> float:
        """
        Get average latency for an operation in milliseconds.
        
        Args:
            name: Operation identifier
            
        Returns:
            Average latency in ms, or 0 if no data
        """
        with self._lock:
            if name not in self._latencies or len(self._latencies[name]) == 0:
                return 0.0
            
            return sum(self._latencies[name]) / len(self._latencies[name])
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics."""
        with self._lock:
            # Calculate total inference latency
            inference_latency = 0.0
            for name in ["water_detection", "obstacle_detection"]:
                if name in self._latencies and self._latencies[name]:
                    inference_latency += sum(self._latencies[name]) / len(self._latencies[name])
            
            # Calculate total pipeline latency
            total_latency = 0.0
            if "pipeline" in self._latencies and self._latencies["pipeline"]:
                total_latency = sum(self._latencies["pipeline"]) / len(self._latencies["pipeline"])
            
            return PerformanceMetrics(
                fps=self.get_fps(),
                inference_latency_ms=inference_latency,
                total_latency_ms=total_latency,
                frame_drops=self._frame_drops,
                memory_mb=0.0  # TODO: Implement memory tracking
            )
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._frame_times.clear()
            self._latencies.clear()
            self._operation_starts.clear()
            self._last_frame_time = None
            self._frame_drops = 0
            self._total_frames = 0


# Global performance monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
