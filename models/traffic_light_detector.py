"""
Traffic light detection — two-stage pipeline.

Stage 1  Locate the traffic-light housing
         Find dark rectangular boxes with portrait aspect ratio (≈ 1:2.5–1:4)
         in the upper portion of the frame.  This dramatically reduces false
         positives from red shirts, green trees, etc.

Stage 2  Classify colour inside each candidate housing
         Divide the ROI into up to three equal horizontal slots (top/mid/bot)
         and decide which slot is brightest in red / yellow / green.
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from utils.logger import get_logger


class TrafficLightColor(Enum):
    """Detected traffic light color."""
    UNKNOWN = "unknown"
    RED     = "red"
    YELLOW  = "yellow"
    GREEN   = "green"


@dataclass
class TrafficLightResult:
    """Result of a single traffic light detection."""
    color: TrafficLightColor = TrafficLightColor.UNKNOWN
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h) in original frame
    center: Optional[Tuple[int, int]] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_go(self) -> bool:
        return self.color == TrafficLightColor.GREEN and self.confidence > 0.5

    @property
    def is_stop(self) -> bool:
        return self.color == TrafficLightColor.RED and self.confidence > 0.5

    @property
    def is_caution(self) -> bool:
        return self.color == TrafficLightColor.YELLOW and self.confidence > 0.45


# ─────────────────────────────────────────────────────────────────────────────
# HSV colour ranges  (Hue in OpenCV = 0-179)
# ─────────────────────────────────────────────────────────────────────────────
_RED_RANGES = [
    (np.array([0,   120, 100], dtype=np.uint8), np.array([10,  255, 255], dtype=np.uint8)),
    (np.array([165, 120, 100], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
]
_YELLOW_RANGE = (
    np.array([18,  100, 120], dtype=np.uint8),
    np.array([32,  255, 255], dtype=np.uint8),
)
_GREEN_RANGE = (
    np.array([45,  80,  80],  dtype=np.uint8),
    np.array([85,  255, 255], dtype=np.uint8),
)


class TrafficLightDetector:
    """
    Two-stage traffic light detector.

    Stage 1 – Housing detection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Look for dark, roughly rectangular regions with a *portrait* aspect ratio
    (height ≈ 2.5–5× width) in the upper portion of the frame.  Traffic-light
    housings are always:
      • taller than wide
      • relatively dark on the outside (black / dark-grey housing)
      • located above the camera horizon (upper 60 % of frame)

    Stage 2 – Colour classification inside the housing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Divide each candidate housing into three equal horizontal thirds
    (top = red zone, middle = yellow zone, bottom = green zone) and test
    how much saturated colour occupies each third.  The slot with the
    highest coloured-pixel ratio wins.

    Additional sanity checks
    ~~~~~~~~~~~~~~~~~~~~~~~~
    • The winning slot must exceed a minimum fill ratio (avoids dim slots).
    • The housing must not be too small (< MIN_HOUSING_AREA) or too large.
    • The best housing is selected by its dark-exterior score.
    """

    # ── Housing geometry constraints ──────────────────────────────────────────
    MIN_HOUSING_AREA  = 300        # px²  — ignore tiny boxes
    MAX_HOUSING_AREA  = 80_000     # px²  — ignore whole-scene boxes
    HOUSING_AR_MIN    = 1.8        # height / width  lower bound
    HOUSING_AR_MAX    = 6.0        # height / width  upper bound

    # ── Colour-slot thresholds ────────────────────────────────────────────────
    MIN_FILL_RATIO    = 0.08       # at least 8 % of a slot must be coloured
    # Confidence is clamped to this max
    MAX_CONFIDENCE    = 0.95

    def __init__(self, scan_top_fraction: float = 0.70):
        """
        Args:
            scan_top_fraction: Only search this fraction of the frame height.
        """
        self.logger = get_logger("traffic_light")
        self.scan_top_fraction = scan_top_fraction
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV not available – TrafficLightDetector disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> TrafficLightResult:
        """
        Detect the most prominent traffic light in *frame*.

        Returns:
            TrafficLightResult  (color=UNKNOWN when nothing convincing found)
        """
        if not CV2_AVAILABLE or frame is None or frame.size == 0:
            return TrafficLightResult()

        try:
            h, w = frame.shape[:2]
            roi_h = int(h * self.scan_top_fraction)
            roi   = frame[:roi_h, :]

            # ── Stage 1: find candidate housings ──────────────────────────────
            housings = self._find_housings(roi)
            if not housings:
                return TrafficLightResult()

            # ── Stage 2: classify colour in each housing ──────────────────────
            best_result: Optional[TrafficLightResult] = None
            best_score  = 0.0

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            for (x, y, bw, bh, housing_score) in housings:
                result = self._classify_colour(hsv, x, y, bw, bh)
                if result is None:
                    continue
                # Combined score = housing quality × colour fill confidence
                combined = housing_score * 0.35 + result.confidence * 0.65
                if combined > best_score:
                    best_score = combined
                    # Remap bbox back to full-frame coordinates (y stays same,
                    # since roi starts at top of frame)
                    best_result = TrafficLightResult(
                        color=result.color,
                        confidence=min(combined, self.MAX_CONFIDENCE),
                        bbox=(x, y, bw, bh),
                        center=(x + bw // 2, y + bh // 2),
                    )

            if best_result is None:
                return TrafficLightResult()

            self.logger.debug(
                f"TL detected: {best_result.color.value}  "
                f"conf={best_result.confidence:.2f}  bbox={best_result.bbox}"
            )
            return best_result

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return TrafficLightResult()

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_housings(self, roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Find dark rectangular boxes with portrait aspect ratio.

        Returns list of (x, y, w, h, score) sorted by score descending.
        Score ∈ [0, 1] — how "housing-like" the region is.
        """
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to isolate dark regions (housing is dark)
        # Use inverse threshold: dark pixels → white in mask
        _, dark_mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        # Also try adaptive threshold for varying lighting
        adapt_mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 8
        )
        combined_mask = cv2.bitwise_or(dark_mask, adapt_mask)

        # Morphological cleanup: close small gaps in the housing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel_open)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_HOUSING_AREA or area > self.MAX_HOUSING_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw == 0:
                continue

            ar = bh / bw  # portrait: should be > 1
            if ar < self.HOUSING_AR_MIN or ar > self.HOUSING_AR_MAX:
                continue

            # Rectangularity: how closely does the contour fill its bounding box?
            rect_fill = area / (bw * bh)
            if rect_fill < 0.35:   # too irregular
                continue

            # Housing score: reward portrait ratio in ideal range & high fill
            ideal_ar_score = 1.0 - abs(ar - 3.0) / 3.0   # peak at ar=3
            ideal_ar_score = max(0.0, ideal_ar_score)
            score = rect_fill * 0.5 + ideal_ar_score * 0.5
            results.append((x, y, bw, bh, float(score)))

        # Sort by score descending, keep top-5 candidates
        results.sort(key=lambda t: t[4], reverse=True)
        return results[:5]

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_colour(
        self,
        hsv: np.ndarray,
        x: int, y: int, bw: int, bh: int,
    ) -> Optional[TrafficLightResult]:
        """
        Decide which light is active inside a housing region.

        The housing is divided into three equal horizontal thirds:
          top third    → red light position
          middle third → yellow light position
          bottom third → green light position

        For each third, we count how many pixels pass the colour mask.
        The third with the highest fill ratio (and above MIN_FILL_RATIO)
        determines the reported colour.

        Returns None if no slot clears the minimum threshold.
        """
        slot_h = bh // 3
        if slot_h < 4:
            return None

        # Extract the three slots from HSV ROI
        slots = {
            TrafficLightColor.RED:    hsv[y:y+slot_h,            x:x+bw],
            TrafficLightColor.YELLOW: hsv[y+slot_h:y+2*slot_h,   x:x+bw],
            TrafficLightColor.GREEN:  hsv[y+2*slot_h:y+3*slot_h, x:x+bw],
        }

        best_color = TrafficLightColor.UNKNOWN
        best_fill  = 0.0

        for color, slot in slots.items():
            if slot.size == 0:
                continue
            mask = self._colour_mask(slot, color)
            fill = float(np.count_nonzero(mask)) / mask.size
            if fill > best_fill:
                best_fill  = fill
                best_color = color

        if best_color == TrafficLightColor.UNKNOWN or best_fill < self.MIN_FILL_RATIO:
            return None

        # Confidence: normalise fill ratio to [0, 1]
        confidence = min(best_fill / 0.35, 1.0)   # 35 % fill → full confidence
        return TrafficLightResult(
            color=best_color,
            confidence=confidence,
        )

    @staticmethod
    def _colour_mask(hsv_slot: np.ndarray, color: TrafficLightColor) -> np.ndarray:
        """Return binary mask of pixels matching *color* in *hsv_slot*."""
        if color == TrafficLightColor.RED:
            m1 = cv2.inRange(hsv_slot, _RED_RANGES[0][0], _RED_RANGES[0][1])
            m2 = cv2.inRange(hsv_slot, _RED_RANGES[1][0], _RED_RANGES[1][1])
            return cv2.bitwise_or(m1, m2)
        elif color == TrafficLightColor.YELLOW:
            return cv2.inRange(hsv_slot, _YELLOW_RANGE[0], _YELLOW_RANGE[1])
        elif color == TrafficLightColor.GREEN:
            return cv2.inRange(hsv_slot, _GREEN_RANGE[0], _GREEN_RANGE[1])
        return np.zeros(hsv_slot.shape[:2], dtype=np.uint8)
