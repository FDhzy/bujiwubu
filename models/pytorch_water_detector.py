"""
水坑检测器 - YOLOv8 segment 任务
替代原有 U-Net PyTorch 模型。
接口与原 PyTorchWaterDetector 完全兼容，WaterDetector 无需改动。
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger


@dataclass
class WaterDetectionResult:
    """Result from water detection."""
    mask: np.ndarray        # 二值 mask，1=水坑 0=非水坑
    confidence: np.ndarray  # 置信度图 (0~1)
    coverage_ratio: float   # 水坑覆盖率
    num_regions: int        # 独立水坑区域数


class PyTorchWaterDetector:
    """
    YOLOv8 segment 水坑检测器。
    接口与原 U-Net 版本完全兼容：
      - detect(frame) -> WaterDetectionResult | None
      - model 属性（None 表示未加载）
    """

    def __init__(self, model_path: str = None, threshold: float = 0.25):
        """
        初始化水坑检测器。

        Args:
            model_path: YOLOv8 segment best.pt 路径
            threshold:  掩码置信度阈值
        """
        self.logger = get_logger("yolov8_water_detector")
        self.model = None          # ultralytics YOLO 实例
        self.threshold = threshold

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            if model_path:
                self.logger.warning(f"水坑模型文件不存在: {model_path}")
            else:
                self.logger.warning("未指定水坑模型路径")

    # ── 加载 ──────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        """加载 YOLOv8 segment .pt 模型。"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.logger.info(f"✓ YOLOv8 水坑分割模型已加载: {model_path}")
        except ImportError:
            self.logger.error("ultralytics 未安装，请执行: pip install ultralytics")
        except Exception as e:
            self.logger.error(f"加载水坑模型失败: {e}")
            self.model = None

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[WaterDetectionResult]:
        """
        检测水坑。

        Args:
            frame: BGR 图像 (HWC 格式，来自 OpenCV)
        Returns:
            WaterDetectionResult 或 None（模型未加载 / 推理出错）
        """
        if self.model is None:
            return None

        try:
            orig_h, orig_w = frame.shape[:2]

            # YOLOv8 segment 推理
            results = self.model(frame, verbose=False)

            # 合并所有分割掩码为单张 confidence map
            confidence = np.zeros((orig_h, orig_w), dtype=np.float32)

            for result in results:
                if result.masks is None:
                    continue

                # result.masks.data: (N, H', W') tensor，值域 [0,1]
                masks_tensor = result.masks.data          # (N, H', W')
                boxes_conf   = result.boxes.conf          # (N,)

                for i in range(len(masks_tensor)):
                    conf_val = float(boxes_conf[i].item())
                    if conf_val < self.threshold:
                        continue

                    # 将掩码 resize 到原始分辨率
                    m = masks_tensor[i].cpu().numpy()     # (H', W')
                    m_resized = cv2.resize(
                        m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                    )
                    # 加权叠加（取最大值，避免重叠区域超过 1）
                    confidence = np.maximum(confidence, m_resized * conf_val)

            # 二值化
            mask = (confidence > self.threshold).astype(np.uint8)

            # 形态学清理：先闭运算填洞，再开运算去噪
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

            # 过滤面积 < 总面积 1% 的碎片区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            min_area = orig_h * orig_w * 0.01
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                    mask[labels == lbl] = 0

            coverage    = float(np.sum(mask)) / float(mask.size)
            num_regions = self._count_regions(mask)

            return WaterDetectionResult(
                mask=mask,
                confidence=confidence,
                coverage_ratio=coverage,
                num_regions=num_regions,
            )

        except Exception as e:
            self.logger.error(f"水坑检测推理失败: {e}")
            return None

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _count_regions(self, mask: np.ndarray) -> int:
        """统计独立水坑区域数量。"""
        n, _ = cv2.connectedComponents(mask)
        return n - 1  # 减去背景

    def estimate_depth(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """
        根据颜色/亮度特征估算相对水深（0~1）。
        保留原接口兼容性。
        """
        if np.sum(mask) == 0:
            return 0.0
        water_pixels = frame[mask > 0]
        if len(water_pixels) == 0:
            return 0.0
        avg_brightness = float(np.mean(water_pixels))
        color_variance = float(np.var(water_pixels))
        depth_est   = (255.0 - avg_brightness) / 255.0
        turbidity   = min(color_variance / 1000.0, 1.0)
        return depth_est * 0.7 + turbidity * 0.3
