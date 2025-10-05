import os, argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image

from models.siamese_change import SiameseUNetResNet50
from utils import load_tensor_rgb, find_pairs
from flood_filter import build_fused_score
from heatmap_vis import save_heatmap_standalone, save_heatmap_overlay

def run_one(pre_path, post_path, args, out_prefix="result"):
    # I/O folders
    os.makedirs(args.out_masks, exist_ok=True)
    os.makedirs(args.out_heatmaps, exist_ok=True)
    os.makedirs(args.out_heatmap_overlays, exist_ok=True)
    os.makedirs(args.out_debug, exist_ok=True)

    # Load tensors
    pre_pil,  pre_t  = load_tensor_rgb(pre_path,  args.size)
    post_pil, post_t = load_tensor_rgb(post_path, args.size)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model  = SiameseUNetResNet50(out_ch=1).to(device).eval()

    with torch.no_grad():
        prob_ = model(pre_t.to(device), post_t.to(device)).squeeze().cpu().numpy()  # [S,S] in [0,1]

    # Prepare fused heat score at SAME resolution as tensors (args.size)
    pre_rgb_s  = np.array(pre_pil.resize((args.size,args.size),  Image.BILINEAR))
    post_rgb_s = np.array(post_pil.resize((args.size,args.size), Image.BILINEAR))
    score = build_fused_score(
        pre_rgb_s, post_rgb_s, prob_,
        w_L=args.w_L, w_edge=args.w_edge, w_prob=args.w_prob,
        use_water_gate=args.use_water_gate,
        blue_minus_green=args.blue_minus_green, max_r=args.max_r,
        p_low=args.p_low, p_high=args.p_high, gamma=args.gamma
    )

    # Output at original size of POST
    post_bgr_full = cv2.cvtColor(np.array(post_pil), cv2.COLOR_RGB2BGR)
    H0, W0 = post_bgr_full.shape[:2]
    score_full = cv2.resize(score, (W0, H0), interpolation=cv2.INTER_LINEAR)

    # Binary masks
    thr_heat = float(np.percentile(score, args.bin_percentile))
    mask_heat = (score > thr_heat).astype(np.uint8)*255
    mask_heat_full = cv2.resize(mask_heat, (W0, H0), interpolation=cv2.INTER_NEAREST)

    mask_model = (prob_ > args.model_thresh).astype(np.uint8)*255
    mask_model_full = cv2.resize(mask_model, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # Save masks
    cv2.imwrite(os.path.join(args.out_masks, f"{out_prefix}_flood_mask.png"), mask_heat_full)
    cv2.imwrite(os.path.join(args.out_masks, f"{out_prefix}_flood_mask_model.png"), mask_model_full)

    # Heatmaps (standalone + overlay)
    save_heatmap_standalone(score_full, os.path.join(args.out_heatmaps, f"{out_prefix}_heatmap.png"))
    save_heatmap_overlay(post_bgr_full, score_full, os.path.join(args.out_heatmap_overlays, f"{out_prefix}_heatmap_overlay.png"), alpha=args.overlay_alpha)

    # Debug: save raw prob resized
    cv2.imwrite(os.path.join(args.out_debug, f"{out_prefix}_change_prob.png"),
                (cv2.resize(prob_, (W0, H0))*255).astype(np.uint8))

def main(args):
    if args.pre and args.post:
        base = os.path.splitext(os.path.basename(args.post))[0]
        run_one(args.pre, args.post, args, out_prefix=base)
        return

    # Directory mode
    pairs = find_pairs(args.pre_dir, args.post_dir, pre_suffix=args.pre_suffix, post_suffix=args.post_suffix)
    if not pairs:
        print("[err] No pairs found."); return
    for pre_path, post_path, base in tqdm(pairs, desc="Pairs"):
        run_one(pre_path, post_path, args, out_prefix=base)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Single-pair or directory mode
    ap.add_argument("--pre", default="", help="path to single pre image (optional)")
    ap.add_argument("--post", default="", help="path to single post image (optional)")
    ap.add_argument("--pre_dir", default="data/pre")
    ap.add_argument("--post_dir", default="data/post")
    ap.add_argument("--pre_suffix", default="_pre")
    ap.add_argument("--post_suffix", default="_post")

    # Output
    ap.add_argument("--out_masks", default="output/masks")
    ap.add_argument("--out_heatmaps", default="output/heatmaps")
    ap.add_argument("--out_heatmap_overlays", default="output/heatmap_overlays")
    ap.add_argument("--out_debug", default="output/debug")

    # Model / processing
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--model_thresh", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")

    # Fusion weights
    ap.add_argument("--w_L",   type=float, default=0.45)
    ap.add_argument("--w_edge",type=float, default=0.25)
    ap.add_argument("--w_prob",type=float, default=0.30)

    # Water prior & contrast/gamma
    ap.add_argument("--use_water_gate", action="store_true")
    ap.add_argument("--blue_minus_green", type=int, default=6)
    ap.add_argument("--max_r", type=int, default=205)
    ap.add_argument("--p_low", type=float, default=2.0)
    ap.add_argument("--p_high", type=float, default=98.0)
    ap.add_argument("--gamma", type=float, default=0.6)

    # Binary mask from fused score
    ap.add_argument("--bin_percentile", type=float, default=80.0)

    # Overlay
    ap.add_argument("--overlay_alpha", type=float, default=0.45)

    args = ap.parse_args()
    main(args)
