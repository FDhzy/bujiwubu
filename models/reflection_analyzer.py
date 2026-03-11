"""
Traditional computer vision water/reflection analyzer.
Uses color analysis and edge detection to supplement ML-based detection.
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger
from utils.config import ReflectionConfig


@dataclass
class ReflectionResult:
    """Result from reflection analysis."""
    # Binary mask where 1 = potential water reflection
    mask: np.ndarray
    # Confidence/intensity map
    intensity: np.ndarray
    # Coverage ratio
    coverage_ratio: float


class ReflectionAnalyzer:
    """
    Traditional CV-based water reflection detector.
    Uses HSV color analysis, edge detection, and texture analysis
    to identify water/reflective surfaces.
    """
    
    def __init__(self, config: ReflectionConfig):
        """
        Initialize reflection analyzer.
        
        Args:
            config: Configuration for reflection detection
        """
        self.logger = get_logger("reflection")
        self.config = config
        
        self.logger.info("ReflectionAnalyzer initialized")
    
    def analyze(self, frame: np.ndarray) -> Optional[ReflectionResult]:
        """
        Analyze frame for water reflections using traditional CV.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            ReflectionResult with mask and metrics, or None on error
        """
        try:
            import cv2
            
            # 1. Preprocessing: Bilateral Filter (Preserves edges, reduces noise better than Gaussian)
            blurred = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            
            # 2. Adaptive HSV-based reflection detection
            # Calculate average brightness to adjust threshold
            avg_v = np.mean(hsv[:, :, 2])
            adaptive_v_thresh = max(self.config.hsv_value_min, avg_v * 1.2)
            hsv_mask = self._detect_hsv_reflections(hsv, adaptive_v_thresh)
            
            # 3. Edge-based water boundary detection (Using Laplacian for better closed loops)
            edge_mask = self._detect_water_edges(gray)
            
            # 4. Improved Texture analysis (Surface Smoothness via local variance)
            texture_mask = self._analyze_texture(gray)
            
            # 5. NEW: Sky/Ground Consistency Check
            # Reflections on water often mirror the color of the upper part of the frame (sky)
            sky_consistency_mask = self._check_sky_reflection(hsv)
            
            # 6. NEW: Flatness & Gradient Analysis
            # Water is horizontal and flat. We look for high vertical gradients relative to horizontal.
            flatness_mask = self._analyze_surface_flatness(gray)
            
            # Combine masks with weights
            combined = (
                0.30 * hsv_mask.astype(np.float32) +
                0.20 * edge_mask.astype(np.float32) +
                0.15 * texture_mask.astype(np.float32) +
                0.15 * sky_consistency_mask.astype(np.float32) +
                0.20 * flatness_mask.astype(np.float32)
            )
            
            # Normalize to intensity map
            intensity = np.clip(combined, 0, 1)
            
            # Threshold to binary mask (Stricter threshold to reduce false positives)
            mask = (intensity > 0.55).astype(np.uint8) 
            
            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.morph_kernel_size, self.config.morph_kernel_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Remove small regions
            mask = self._remove_small_regions(mask)
            
            coverage = np.sum(mask) / mask.size
            
            return ReflectionResult(
                mask=mask,
                intensity=intensity,
                coverage_ratio=coverage
            )
            
        except Exception as e:
            self.logger.error(f"Reflection analysis error: {e}")
            return None
    
    def _analyze_surface_flatness(self, gray: np.ndarray) -> np.ndarray:
        """
        Analyze the surface flatness by comparing vertical and horizontal gradients.
        Water puddles are horizontal and have very low vertical internal gradients.
        """
        import cv2
        # Calculate Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # In a puddle, we expect some horizontal gradients (reflections) 
        # but very little vertical structure.
        mag_x = cv2.absdiff(grad_x, 0)
        mag_y = cv2.absdiff(grad_y, 0)
        
        # Calculate ratio of horizontal to vertical gradient
        # Higher ratio = more likely a flat surface reflecting things
        ratio = (mag_x + 1.0) / (mag_y + 1.0)
        cv2.normalize(ratio, ratio, 0, 1, cv2.NORM_MINMAX)
        
        # Threshold for high horizontal/vertical ratio
        flat_mask = (ratio > 0.6).astype(np.uint8)
        
        # Ground focus
        h, w = flat_mask.shape
        flat_mask[:int(h * 0.4), :] = 0
        return flat_mask

    def _check_sky_reflection(self, hsv: np.ndarray) -> np.ndarray:
        """Check if bottom regions mirror the color distribution of the top region."""
        import cv2
        h, w = hsv.shape[:2]
        # Sample sky region (top 20%)
        sky_roi = hsv[0:int(h*0.2), :, :]
        sky_hist_h = cv2.calcHist([sky_roi], [0], None, [180], [0, 180])
        cv2.normalize(sky_hist_h, sky_hist_h, 0, 255, cv2.NORM_MINMAX)
        
        # Backproject sky color onto the whole image
        backproj = cv2.calcBackProject([hsv], [0], sky_hist_h, [0, 180], 1)
        
        # Mask out sky itself
        mask = (backproj / 255.0).astype(np.float32)
        mask[0:int(h*0.4), :] = 0  # Focus on ground
        return mask

    def _detect_hsv_reflections(self, hsv: np.ndarray, v_thresh: float) -> np.ndarray:
        """
        Detect reflective surfaces using HSV color space.
        """
        import cv2
        
        # Extract channels
        h, s, v = cv2.split(hsv)
        
        # High brightness regions (reflections)
        bright_mask = v > v_thresh
        
        # Low saturation often indicates sky/cloud reflections on water
        sat_mask = (s < 80)
        
        # Combined: bright regions with low saturation
        mask = (bright_mask & sat_mask).astype(np.uint8)
        
        # Ground focus: Ignore top 40% (more aggressive than 30%)
        h, w = mask.shape
        mask[:int(h * 0.4), :] = 0
        
        return mask
    
    def _detect_water_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect water boundaries using edge detection.
        Water puddles often have distinct edges where they meet dry surfaces.
        """
        import cv2
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2
        )
        
        # Dilate edges to create regions
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours and fill
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.min_contour_area:
                cv2.fillPoly(mask, [contour], 255)
        
        return (mask > 0).astype(np.uint8)
    
    def _analyze_texture(self, gray: np.ndarray) -> np.ndarray:
        """
        Analyze texture to find smooth surfaces (potential water).
        Water surfaces are typically smoother than surrounding textures.
        """
        import cv2
        
        # Calculate local variance as texture measure
        # Low variance = smooth surface (potential water)
        
        # Apply local standard deviation filter
        kernel_size = 15
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        variance = np.maximum(variance, 0)  # Ensure non-negative
        std_dev = np.sqrt(variance)
        
        # Normalize
        std_dev_normalized = std_dev / (std_dev.max() + 1e-6)
        
        # Low texture (smooth) regions are potential water
        # Invert so that smooth = high value
        smooth_mask = 1.0 - std_dev_normalized
        
        # Threshold
        mask = (smooth_mask > 0.7).astype(np.uint8)
        
        # Focus on lower portion of frame
        h, w = mask.shape
        mask[:int(h * 0.4), :] = 0
        
        return mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected components from mask."""
        import cv2
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        result = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.min_contour_area:
                result[labels == i] = 1
        
        return result
    
    def combine_with_ml(
        self,
        ml_mask: np.ndarray,
        reflection_result: ReflectionResult,
        ml_weight: float = 0.7
    ) -> np.ndarray:
        """
        Combine ML-based detection with traditional CV results.
        
        Args:
            ml_mask: Binary mask from ML model
            reflection_result: Result from reflection analysis
            ml_weight: Weight for ML results (0-1)
            
        Returns:
            Combined binary mask
        """
        import cv2
        
        # Resize reflection mask to match ML mask if needed
        if reflection_result.mask.shape != ml_mask.shape:
            reflection_mask = cv2.resize(
                reflection_result.mask.astype(np.float32),
                (ml_mask.shape[1], ml_mask.shape[0])
            )
        else:
            reflection_mask = reflection_result.mask.astype(np.float32)
        
        ml_mask_float = ml_mask.astype(np.float32)
        
        # Weighted combination
        combined = (
            ml_weight * ml_mask_float +
            (1 - ml_weight) * reflection_mask
        )
        
        # Use OR logic for final mask (union of both detections)
        # This ensures we don't miss water detected by either method
        final_mask = ((ml_mask > 0) | (reflection_result.mask > 0)).astype(np.uint8)
        
        return final_mask
