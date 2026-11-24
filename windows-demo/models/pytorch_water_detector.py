"""
水坑检测器 - 基于 TFLite U-Net
使用 training_checkpoints/water_model.tflite
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TFLite not available for water detection")


@dataclass
class WaterDetectionResult:
    """Result from water detection."""
    mask: np.ndarray
    confidence: np.ndarray
    coverage_ratio: float
    num_regions: int


class PyTorchWaterDetector:
    """
    水坑检测器 - 使用 TFLite U-Net 推理
    """

    INPUT_SIZE = 256  # U-Net 输入尺寸
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str = None, threshold: float = 0.15):
        """
        初始化水坑检测器

        Args:
            model_path: TFLite 模型路径
            threshold: 检测阈值
        """
        self.logger = get_logger("pytorch_water_detector")
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.threshold = threshold

        if not TFLITE_AVAILABLE:
            self.logger.error("TFLite not available")
            return

        # 默认路径
        if model_path is None:
            model_path = "training_checkpoints/water_model.tflite"

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载 TFLite 模型"""
        try:
            path = Path(model_path)
            if not path.exists():
                self.logger.warning(f"模型不存在: {path}")
                return
            self.interpreter = tflite.Interpreter(model_content=path.read_bytes())
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info(f"TFLite 水坑检测器已加载: {path}")
        except Exception as e:
            self.logger.error(f"加载 TFLite 水坑模型失败: {e}")
            self.interpreter = None

    def detect(self, frame: np.ndarray) -> Optional[WaterDetectionResult]:
        """
        检测水坑

        Args:
            frame: BGR图像 (HWC格式)

        Returns:
            WaterDetectionResult or None
        """
        if self.interpreter is None:
            return None

        try:
            original_h, original_w = frame.shape[:2]

            # 预处理: BGR→RGB, resize, 归一化, CHW, batch
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            img_float = img_resized.astype(np.float32) / 255.0
            img_norm = (img_float - self.MEAN) / self.STD
            # HWC → CHW → batch: [1, 3, 256, 256]
            input_data = img_norm[None, ...]

            # TFLite 推理
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            # output shape: [1, 1, 256, 256]
            confidence = output[0, :, :, 0]
            if output.shape != (1, self.INPUT_SIZE, self.INPUT_SIZE, 1) or not np.isfinite(confidence).all():
                raise ValueError("Unexpected water model output")  # shape [256, 256]
            
            # 调整大小到原始尺寸
            confidence = cv2.resize(confidence, (original_w, original_h))
            
            # 生成mask
            mask = (confidence > self.threshold).astype(np.uint8)
            
            # 形态学操作清理
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 过滤小面积噪点（小于总面积1%的区域视为误报）
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            min_area = original_h * original_w * 0.01
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < min_area:
                    mask[labels == label] = 0
            
            # 计算指标
            coverage = np.sum(mask) / mask.size
            num_regions = self._count_regions(mask)
            
            return WaterDetectionResult(
                mask=mask,
                confidence=confidence,
                coverage_ratio=coverage,
                num_regions=num_regions
            )
            
        except Exception as e:
            self.logger.error(f"Water detection error: {e}")
            return None
    
    def _count_regions(self, mask: np.ndarray) -> int:
        """计算水坑区域数量"""
        num_labels, _ = cv2.connectedComponents(mask)
        return num_labels - 1
    
    def estimate_depth(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """
        估算水深（简化版）
        
        基于颜色和纹理特征估算相对水深
        """
        if np.sum(mask) == 0:
            return 0.0
        
        # 提取水坑区域
        water_region = frame[mask > 0]
        
        if len(water_region) == 0:
            return 0.0
        
        # 计算平均亮度（暗的水坑通常更深）
        avg_brightness = np.mean(water_region)
        
        # 计算颜色方差（浑浊的水方差更大）
        color_variance = np.var(water_region)
        
        # 综合估算（简化算法）
        # 越暗越可能深，但浑浊度也要考虑
        depth_estimate = (255 - avg_brightness) / 255.0
        turbidity = min(color_variance / 1000.0, 1.0)
        
        # 综合得分
        depth_score = depth_estimate * 0.7 + turbidity * 0.3
        
        return float(depth_score)
