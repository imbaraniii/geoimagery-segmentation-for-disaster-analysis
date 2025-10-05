import numpy as np
import cv2

def normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

def to_lab_l(rgb_uint8):
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    L   = lab[...,0].astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(L.astype(np.uint8)).astype(np.float32)

def edge_mag(rgb_uint8):
    g  = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g  = cv2.GaussianBlur(g, (0,0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)

def water_prior_rgb(rgb_uint8, blue_minus_green=6, max_r=205):
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR).astype(np.float32)
    b,g,r = cv2.split(bgr)
    bd = np.clip((b - g - blue_minus_green)/50.0, 0.0, 1.0)   # blue dominance
    lr = np.clip((max_r - r)/80.0, 0.0, 1.0)                  # low red
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0,0), 1.2)
    var  = cv2.GaussianBlur((gray - blur)**2, (0,0), 1.2)
    vt   = 1.0 - (var/(var.mean()*3.0 + 1e-6))                # smoothness
    vt   = np.clip(vt, 0.0, 1.0)
    return np.clip(0.5*bd + 0.3*lr + 0.2*vt, 0.0, 1.0)

def contrast_stretch01(x, p_low=2.0, p_high=98.0, eps=1e-6):
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

def gamma_boost(x01, gamma=0.6):
    return np.power(np.clip(x01, 0, 1), gamma)

def build_fused_score(pre_rgb, post_rgb, prob_01,
                      w_L=0.45, w_edge=0.25, w_prob=0.30,
                      use_water_gate=True, blue_minus_green=6, max_r=205,
                      p_low=2.0, p_high=98.0, gamma=0.6):
    """All inputs at SAME size (HxW). Returns score in [0,1]."""
    # ensure same resolution
    H, W = post_rgb.shape[:2]
    if pre_rgb.shape[:2] != (H,W):
        pre_rgb = cv2.resize(pre_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    if prob_01.shape[:2] != (H,W):
        prob_01 = cv2.resize(prob_01, (W, H), interpolation=cv2.INTER_LINEAR)

    L_pre  = to_lab_l(pre_rgb)
    L_post = to_lab_l(post_rgb)
    diff_L = normalize01(np.abs(L_post - L_pre))

    e_pre  = edge_mag(pre_rgb)
    e_post = edge_mag(post_rgb)
    e_diff = normalize01(np.abs(e_post - e_pre))

    score = w_L*diff_L + w_edge*e_diff + w_prob*np.clip(prob_01,0,1)

    if use_water_gate:
        water = water_prior_rgb(post_rgb, blue_minus_green=blue_minus_green, max_r=max_r)
        score = score * np.clip(water, 0, 1)

    score = contrast_stretch01(score, p_low=p_low, p_high=p_high)
    score = gamma_boost(score, gamma=gamma)
    return np.clip(score, 0, 1)
