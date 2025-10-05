# flood_gate.py
import numpy as np
import cv2

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def to_prob(arr):
    arr = np.asarray(arr, dtype="float32")
    if 0.0 <= arr.min() and arr.max() <= 1.0:
        return arr
    return sigmoid(arr)

def auto_threshold(prob, fallback=0.35):
    """
    Robust threshold:
    - Try Otsu on prob*255.
    - If Otsu too low/high (degenerate), fall back to percentile or fixed.
    """
    p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(p8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = otsu / 255.0
    if t < 0.05 or t > 0.9:
        # percentile-based fallback
        t = float(np.percentile(prob, 85))
        if t < 0.15 or t > 0.9:
            t = fallback
    return t

def clean(mask_u8, min_area=120):
    """
    Remove tiny blobs and close small gaps.
    """
    nb, lbl, stats, _ = cv2.connectedComponentsWithStats((mask_u8>0).astype(np.uint8), 8)
    out = np.zeros_like(mask_u8)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    return out

def water_gate_rgb(post_bgr, mask_u8, blue_minus_green=10):
    """
    Keep only regions that look water-like in RGB (blue a bit stronger than green).
    Helps when no NIR band is available.
    """
    b, g, r = cv2.split(post_bgr.astype(np.float32))
    water_like = ((b - g) > blue_minus_green).astype(np.uint8) * 255
    keep = ((mask_u8 > 0) & (water_like > 0)).astype(np.uint8) * 255
    return keep

def overlay(post_bgr, mask_u8, alpha=0.4):
    """
    Red overlay on top of post image.
    """
    post = post_bgr.copy()
    overlay = post.copy()
    overlay[mask_u8 > 0] = (0, 0, 255) # red in BGR
    out = (post * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return out
