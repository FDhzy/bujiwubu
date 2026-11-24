"""Local Windows demo detector; not validated for real-world navigation."""
from pathlib import Path
import cv2
import numpy as np
from models.water_detector import WaterDetectionResult

class DesktopWaterDetector:
    threshold = 0.60
    min_area_fraction = 0.01

    def __init__(self, model_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(str(Path(model_path).resolve()), compile=False, safe_mode=True)
        if self.model.input_shape[1:] != (256, 256, 3) or self.model.output_shape[1:] != (256, 256, 1):
            raise ValueError("Unexpected archived water model dimensions")

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
        y = self.model(x[None, ...], training=False).numpy()[0, :, :, 0]
        if not np.isfinite(y).all():
            raise ValueError("Non-finite water prediction")
        confidence = cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (confidence > self.threshold).astype(np.uint8)
        k = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        regions = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_fraction * mask.size:
                mask[labels == i] = 0
            else:
                regions += 1
        return WaterDetectionResult(mask=mask, confidence=confidence,
                                    coverage_ratio=float(mask.mean()), num_regions=regions)
