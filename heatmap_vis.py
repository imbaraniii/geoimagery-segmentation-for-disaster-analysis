import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_heatmap_standalone(score01, out_png, title="Heatmap: Blue (Non-affected) → Red (Affected)"):
    h, w = score01.shape
    dpi = 140
    fig_w, fig_h = max(4, w/dpi), max(4, h/dpi)
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    im = plt.imshow(score01, cmap="coolwarm", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = plt.colorbar(im); cbar.set_label("Difference Intensity")
    plt.title(title); plt.axis("off"); plt.tight_layout(pad=0.2)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1); plt.close()

def save_heatmap_overlay(post_bgr, score01, out_png, alpha=0.45):
    H, W = post_bgr.shape[:2]
    if score01.shape != (H, W):
        score01 = cv2.resize(score01, (W, H), interpolation=cv2.INTER_LINEAR)
    cmap = plt.get_cmap("coolwarm")
    heat_rgb = (cmap(np.clip(score01, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    blend = (post_bgr.astype(np.float32)*(1-alpha) + heat_bgr.astype(np.float32)*alpha).astype(np.uint8)
    cv2.imwrite(out_png, blend)
