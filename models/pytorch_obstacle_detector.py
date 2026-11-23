"""
PyTorch 障碍物检测器
支持加载 YOLOv8 训练的 .pt 权重文件
使用 training_checkpoints/obstacle/best.pt
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """检测结果"""
    class_id: int
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


class PyTorchObstacleDetector:
    """YOLOv8 障碍物检测器 - 使用 ultralytics 加载训练好的 .pt 权重"""

    # 与训练数据集 dataset.yaml 中的 names 完全对应（11个类别）
    CATEGORIES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'stairs',
        5: 'wire',
        6: 'bench',
        7: 'speed_bump',
        8: 'pothole',
        9: 'utility_pole',
        10: 'tree',
    }

    def __init__(self, model_path: str = "training_checkpoints/obstacle/best.pt",
                 conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)

        if self.model is not None:
            print(f"✓ YOLOv8 障碍物检测器已加载: {model_path}")
            print(f"  类别数: {len(self.CATEGORIES)}")
        else:
            print("✗ 模型加载失败")

    def _load_model(self, model_path: str):
        """加载 YOLOv8 .pt 模型"""
        try:
            from ultralytics import YOLO
            path = Path(model_path)
            if not path.exists():
                print(f"模型不存在: {path}")
                return None
            model = YOLO(str(path))
            return model
        except Exception as e:
            print(f"加载 YOLOv8 模型失败: {e}")
            return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """检测障碍物，返回 Detection 列表"""
        if self.model is None:
            return []

        try:
            orig_h, orig_w = frame.shape[:2]
            # ultralytics 推理，verbose=False 避免刷屏
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = []

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    # xyxy 格式转换为 xywh
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x = max(0, int(x1))
                    y = max(0, int(y1))
                    w = min(int(x2 - x1), orig_w - x)
                    h = min(int(y2 - y1), orig_h - y)

                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=self.CATEGORIES.get(cls_id, f'class_{cls_id}'),
                        confidence=conf,
                        x=x,
                        y=y,
                        width=w,
                        height=h
                    ))

            return detections

        except Exception as e:
            print(f"YOLOv8 推理失败: {e}")
            return []


def test_detector():
    """测试检测器"""
    print("=" * 70)
    print("测试 PyTorch 障碍物检测器")
    print("=" * 70)
    
    detector = PyTorchObstacleDetector()
    
    if detector.model is None:
        print("\n✗ 模型加载失败，无法测试")
        return
    
    # 创建测试图像
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(test_img, (300, 150), (340, 380), (100, 100, 100), -1)
    
    print("\n测试检测...")
    detections = detector.detect(test_img)
    
    print(f"检测到 {len(detections)} 个障碍物")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} at ({det.x}, {det.y})")


if __name__ == "__main__":
    test_detector()
