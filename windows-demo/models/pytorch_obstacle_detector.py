"""
PyTorch 障碍物检测器
支持加载 TFLite 转换后的 YOLOv8 权重文件
使用 training_checkpoints/obstacle_best.tflite
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

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
        print("Warning: TFLite not available for obstacle detection")


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
    """YOLOv8 障碍物检测器 - 使用 TFLite 推理"""

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

    # YOLOv8 输入尺寸
    INPUT_SIZE = 320

    def __init__(self, model_path: str = "training_checkpoints/obstacle_best.tflite",
                 conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model(model_path)

        if self.interpreter is not None:
            print(f"✓ TFLite 障碍物检测器已加载: {model_path}")
            print(f"  类别数: {len(self.CATEGORIES)}")
        else:
            print("✗ 模型加载失败")

    def _load_model(self, model_path: str):
        """加载 TFLite 模型"""
        if not TFLITE_AVAILABLE:
            print("TFLite 不可用，无法加载障碍物模型")
            return
        try:
            path = Path(model_path)
            if not path.exists():
                print(f"模型不存在: {path}")
                return
            self.interpreter = tflite.Interpreter(model_content=path.read_bytes())
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"加载 TFLite 障碍物模型失败: {e}")
            self.interpreter = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """预处理图像为 YOLOv8 输入格式 [1,3,640,640] float32"""
        img = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # CHW 格式
        return np.expand_dims(img, axis=0)

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> List[Detection]:
        """
        解析 YOLOv8 输出: shape [1, num_classes+4, num_anchors]
        格式: [cx, cy, w, h, cls0_score, cls1_score, ...]
        """
        # output shape: [1, 15, 2100] => squeeze to [15, 2100]
        pred = output[0]  # shape: [15, num_anchors] 或 [num_anchors, 15]
        # 确保维度正确: 如果 shape[0] 比 shape[1] 大，则转置
        columns = len(self.CATEGORIES) + 4
        if pred.shape[0] == columns:
            pred = pred.T
        if pred.shape[1] != columns or not np.isfinite(pred).all():
            raise ValueError("Unexpected obstacle output")  # => [num_anchors, 15]

        num_classes = len(self.CATEGORIES)
        # pred: [num_anchors, 4 + num_classes]
        boxes_cxcywh = pred[:, :4]  # cx, cy, w, h (归一化到640)
        class_scores = pred[:, 4:]  # [num_anchors, num_classes]

        # 每个anchor取最高类别分数
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 置信度过滤
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids = class_ids[mask]
        scores = max_scores[mask]

        # cx,cy,w,h → x1,y1,x2,y2 (归一化)
        scale_x = orig_w
        scale_y = orig_h
        cx = boxes_cxcywh[:, 0] * scale_x
        cy = boxes_cxcywh[:, 1] * scale_y
        bw = boxes_cxcywh[:, 2] * scale_x
        bh = boxes_cxcywh[:, 3] * scale_y
        x1 = (cx - bw / 2).astype(int)
        y1 = (cy - bh / 2).astype(int)
        x2 = (cx + bw / 2).astype(int)
        y2 = (cy + bh / 2).astype(int)

        x1, x2 = np.clip(x1, 0, orig_w), np.clip(x2, 0, orig_w)
        y1, y2 = np.clip(y1, 0, orig_h), np.clip(y2, 0, orig_h)
        boxes_nms = np.stack([x1, y1, x2-x1, y2-y1], axis=1).astype(float)
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(),
            self.conf_threshold, 0.45
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()

        detections = []
        for i in indices:
            x = max(0, int(x1[i]))
            y = max(0, int(y1[i]))
            w = min(int(x2[i] - x1[i]), orig_w - x)
            h = min(int(y2[i] - y1[i]), orig_h - y)
            if w <= 0 or h <= 0:
                continue
            cls_id = int(class_ids[i])
            detections.append(Detection(
                class_id=cls_id,
                class_name=self.CATEGORIES.get(cls_id, f'class_{cls_id}'),
                confidence=float(scores[i]),
                x=x, y=y, width=w, height=h
            ))
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """检测障碍物，返回 Detection 列表"""
        if self.interpreter is None:
            raise RuntimeError("Obstacle model is not available")

        try:
            orig_h, orig_w = frame.shape[:2]
            input_data = self._preprocess(frame)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            return self._postprocess(output, orig_w, orig_h)

        except Exception as e:
            print(f"TFLite 障碍物推理失败: {e}")
            raise


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
