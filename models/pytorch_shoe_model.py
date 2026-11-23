"""
YOLOv8-based Shoe Classification Model
使用 YOLOv8 classify 任务训练的 best.pt 进行鞋型分类。
接口与原 PyTorchShoeModel 完全兼容，无需修改上层调用代码。
"""
import os
import hashlib
import numpy as np
import cv2
from typing import Tuple, Optional, List

from models.shoe_analyzer import ShoeType, ExternalShoeModel
from models.shoe_database import ShoeDatabase
from utils.logger import get_logger

# ── 类别名称 → ShoeType 映射表 ────────────────────────────────────────────────
# YOLOv8 classify 训练时文件夹名即类别名，与此表对应即可
_NAME_TO_TYPE = {
    # 凉鞋 / 拖鞋
    "sandals":     ShoeType.SANDALS,
    "sandal":      ShoeType.SANDALS,
    "slipper":     ShoeType.SANDALS,
    "flip_flop":   ShoeType.SANDALS,
    # 低帮鞋 / 运动鞋
    "low_shoes":   ShoeType.LOW_SHOES,
    "sneakers":    ShoeType.LOW_SHOES,
    "sneaker":     ShoeType.LOW_SHOES,
    "shoes":       ShoeType.LOW_SHOES,
    "shoe":        ShoeType.LOW_SHOES,
    # 踝靴
    "ankle_shoes": ShoeType.ANKLE_SHOES,
    "ankle_boot":  ShoeType.ANKLE_SHOES,
    # 中筒靴
    "boots":       ShoeType.BOOTS,
    "boot":        ShoeType.BOOTS,
    # 高筒靴 / 雨靴
    "high_boots":  ShoeType.HIGH_BOOTS,
    "rain_boots":  ShoeType.HIGH_BOOTS,
    "rain_boot":   ShoeType.HIGH_BOOTS,
    "wellington":  ShoeType.HIGH_BOOTS,
}

# 默认 3 类索引映射（训练数据未覆盖所有类时的兜底）
_DEFAULT_IDX_MAP = {
    0: ShoeType.SANDALS,
    1: ShoeType.LOW_SHOES,
    2: ShoeType.BOOTS,
}


class YOLOv8ShoeModel(ExternalShoeModel):
    """
    YOLOv8 classify 鞋型识别模型。
    接口与原 PyTorchShoeModel 完全兼容：
      - load_model(model_path) -> bool
      - predict(image) -> (ShoeType, confidence)
    """

    def __init__(self, use_database: bool = True):
        self.logger = get_logger("yolov8_shoe_model")
        self.model = None           # ultralytics YOLO 实例
        self.names: List[str] = []  # 类别名列表（从模型读取）
        self._idx_map: dict = {}    # index -> ShoeType
        self.conf_threshold: float = 0.15
        self.use_database = use_database
        self.database: Optional[ShoeDatabase] = None

        if use_database:
            try:
                self.database = ShoeDatabase()
            except Exception as e:
                self.logger.warning(f"数据库连接失败: {e}")
                self.use_database = False

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> bool:
        """
        加载 YOLOv8 classify .pt 权重。

        Args:
            model_path: best.pt 路径
        Returns:
            True 表示加载成功
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.error("ultralytics 未安装，请执行: pip install ultralytics")
            return False

        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return False

        try:
            self.model = YOLO(model_path)
            # 读取类别列表
            if hasattr(self.model, 'names') and self.model.names:
                # model.names 可能是 dict {0:'sandals',...} 或 list
                if isinstance(self.model.names, dict):
                    self.names = [self.model.names[i]
                                  for i in sorted(self.model.names.keys())]
                else:
                    self.names = list(self.model.names)
            self._build_idx_map()
            self.logger.info(
                f"YOLOv8 鞋型模型加载成功: {model_path}\n"
                f"  类别({len(self.names)}): {self.names}\n"
                f"  索引映射: {self._idx_map}"
            )
            return True
        except Exception as e:
            self.logger.error(f"加载 YOLOv8 鞋型模型失败: {e}")
            return False

    def predict(self, image: np.ndarray) -> Tuple[ShoeType, float]:
        """
        对输入图像进行鞋型分类。

        Args:
            image: BGR numpy 数组 (来自 OpenCV)
        Returns:
            (ShoeType, confidence)  confidence ∈ [0, 1]
        """
        if self.model is None:
            self.logger.warning("模型未加载")
            return ShoeType.UNKNOWN, 0.0

        try:
            # ── 缓存查询 ──────────────────────────────────────────────────
            image_hash = None
            if self.use_database and self.database:
                image_hash = self._hash(image)
                cached = self.database.get_from_cache(image_hash)
                if cached:
                    shoe_type_str, conf = cached
                    return ShoeType(shoe_type_str), conf

            # ── 裁剪鞋子区域 ──────────────────────────────────────────────
            cropped = self._crop_shoe_region(image)

            # ── YOLOv8 推理 ───────────────────────────────────────────────
            results = self.model(cropped, verbose=False)
            if not results:
                return ShoeType.UNKNOWN, 0.0

            res = results[0]
            # classify 结果在 res.probs
            if res.probs is None:
                return ShoeType.UNKNOWN, 0.0

            probs = res.probs.data.cpu().numpy()   # shape: (num_classes,)
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])

            # 调试日志
            for i, p in enumerate(probs):
                name = self.names[i] if i < len(self.names) else f"class_{i}"
                self.logger.debug(f"  [{i}] {name}: {p:.2%}")

            if confidence < self.conf_threshold:
                return ShoeType.UNKNOWN, confidence

            shoe_type = self._idx_map.get(predicted_idx, ShoeType.UNKNOWN)
            self.logger.info(
                f"鞋型识别: {shoe_type.value}  置信度={confidence:.2%}  "
                f"(类别索引={predicted_idx}, 名称="
                f"{self.names[predicted_idx] if predicted_idx < len(self.names) else '?'})"
            )

            # ── 写入缓存 ──────────────────────────────────────────────────
            if self.use_database and self.database and image_hash:
                self.database.save_recognition(shoe_type.value, confidence,
                                               image_hash=image_hash)

            return shoe_type, confidence

        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return ShoeType.UNKNOWN, 0.0

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _build_idx_map(self):
        """根据 self.names 构建 index -> ShoeType 映射。"""
        if self.names:
            self._idx_map = {}
            for i, name in enumerate(self.names):
                shoe_type = _NAME_TO_TYPE.get(name.lower(), ShoeType.UNKNOWN)
                self._idx_map[i] = shoe_type
        else:
            self._idx_map = dict(_DEFAULT_IDX_MAP)

    def _crop_shoe_region(self, image: np.ndarray) -> np.ndarray:
        """
        取图像下 2/3，用边缘检测定位最大前景轮廓，裁剪鞋子区域。
        找不到合适轮廓时直接返回下 2/3 区域。
        """
        try:
            h, w = image.shape[:2]
            lower = image[h // 3:, :]
            lh, lw = lower.shape[:2]

            gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > lh * lw * 0.05:
                    x, y, cw, ch = cv2.boundingRect(c)
                    pad = 10
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(lw, x + cw + pad), min(lh, y + ch + pad)
                    cropped = lower[y1:y2, x1:x2]
                    if cropped.size > 0:
                        return cropped
            return lower
        except Exception:
            return image

    def _hash(self, image: np.ndarray) -> str:
        """计算图像指纹（用于缓存键）。"""
        small = cv2.resize(image, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()

    def get_statistics(self):
        """兼容原接口：从数据库获取统计信息。"""
        if self.database:
            return self.database.get_statistics()
        return {'total': 0}

    def close(self):
        """释放资源。"""
        if self.database:
            self.database.close()

    def __del__(self):
        self.close()


# ── 向后兼容别名 ──────────────────────────────────────────────────────────────
# 旧代码 import PyTorchShoeModel 仍可正常使用
PyTorchShoeModel = YOLOv8ShoeModel
