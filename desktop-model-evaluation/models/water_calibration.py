"""Experimental Windows puddle calibration; not a navigation safety system."""
import cv2
import numpy as np

def image_input(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (256, 256)).astype(np.float32) / 255

def features(rgb, y, flip_y):
    # Pixel-level calibrator. Position agnostic; only appearance and model responses.
    mid = (y + flip_y) * .5
    smooth = cv2.GaussianBlur(mid, (0, 0), 6)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean = cv2.GaussianBlur(gray, (0, 0), 3)
    std = np.sqrt(np.maximum(0, cv2.GaussianBlur(gray*gray, (0, 0), 3) - mean*mean))
    return np.stack((mid, np.abs(y-flip_y), smooth, mid-smooth,
                     hsv[:,:,1], gray, std), axis=-1)

def project(f, params):
    z = np.einsum('hwk,k->hw', f, np.asarray(params['coef'],dtype=np.float32)) + params['bias']
    return 1 / (1 + np.exp(-np.clip(z,-30,30)))

def postprocess(confidence, threshold, min_area_fraction=.01):
    mask = (confidence > threshold).astype(np.uint8)
    k = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = stats[:,cv2.CC_STAT_AREA] >= min_area_fraction * mask.size
    keep[0] = False
    return keep[labels].astype(np.uint8), int(keep.sum())

