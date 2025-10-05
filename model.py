import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# -------- torchvision imports with version fallback --------
from torchvision import transforms
def _get_resnet50_backbone():
    from torchvision.models import resnet50
    try:
        # torchvision >= 0.13
        from torchvision.models import ResNet50_Weights
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        return resnet50(pretrained=True)

# ---------------- Encoder: ResNet50 ----------------
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = _get_resnet50_backbone()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.layer0(x)     # [B,64,H/4,W/4]
        x1 = self.layer1(x0)    # [B,256,H/4,W/4]
        x2 = self.layer2(x1)    # [B,512,H/8,W/8]
        x3 = self.layer3(x2)    # [B,1024,H/16,W/16]
        x4 = self.layer4(x3)    # [B,2048,H/32,W/32]
        return x0, x1, x2, x3, x4

# ---------------- U-Net style decoder block ----------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        return x

# ---------------- Siamese U-Net (ResNet50) ----------------
class SiameseUNetResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.dec4 = DecoderBlock(2048, 1024*2, 512)
        self.dec3 = DecoderBlock(512, 512*2, 256)
        self.dec2 = DecoderBlock(256, 256*2, 128)
        self.dec1 = DecoderBlock(128, 64*2, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, pre, post):
        p0,p1,p2,p3,p4 = self.encoder(pre)
        q0,q1,q2,q3,q4 = self.encoder(post)
        d4 = torch.abs(q4 - p4)
        x  = self.dec4(d4, torch.cat([p3,q3], dim=1))
        x  = self.dec3(x,  torch.cat([p2,q2], dim=1))
        x  = self.dec2(x,  torch.cat([p1,q1], dim=1))
        x  = self.dec1(x,  torch.cat([p0,q0], dim=1))
        prob = torch.sigmoid(self.out(x))      # [B,1,H,W] in [0,1]
        return prob

# ---------------- Preprocess ----------------
def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_image_tensor(path, size):
    pil = Image.open(path).convert("RGB")
    return pil, build_transform(size)(pil).unsqueeze(0)

def normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

# ---------------- Stronger Heatmap Cues ----------------
def to_lab_l(img_rgb_uint8):
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    L   = lab[...,0].astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    L_eq = clahe.apply(L.astype(np.uint8)).astype(np.float32)
    return L_eq

def edge_mag_gray(img_rgb_uint8):
    g = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0,0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)

def contrast_stretch01(x, p_low=2.0, p_high=98.0, eps=1e-6):
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

def gamma_boost(x01, gamma=0.6):
    return np.power(np.clip(x01, 0, 1), gamma)

def water_prior_rgb(post_rgb_uint8, blue_minus_green=6, max_r=205):
    bgr = cv2.cvtColor(post_rgb_uint8, cv2.COLOR_RGB2BGR).astype(np.float32)
    b,g,r = cv2.split(bgr)
    bd = np.clip((b - g - blue_minus_green)/50.0, 0.0, 1.0)     # blue dominance
    lr = np.clip((max_r - r)/80.0, 0.0, 1.0)                    # low red
    gray = cv2.cvtColor(post_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0,0), 1.2)
    var  = cv2.GaussianBlur((gray - blur)**2, (0,0), 1.2)
    vt   = 1.0 - (var/(var.mean()*3.0 + 1e-6))                  # smoothness
    vt   = np.clip(vt, 0.0, 1.0)
    return np.clip(0.5*bd + 0.3*lr + 0.2*vt, 0.0, 1.0)

# ---------------- Heatmap Writers ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_heatmap_standalone(score01, out_png, title="Heatmap: Blue (Non-affected) → Red (Affected)"):
    h, w = score01.shape
    dpi = 140
    fig_w, fig_h = max(4, w/dpi), max(4, h/dpi)
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    im = plt.imshow(score01, cmap="coolwarm", vmin=0.0, vmax=1.0, interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label("Difference Intensity")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def save_heatmap_overlay(post_bgr, score01, out_png, alpha=0.45):
    H, W = post_bgr.shape[:2]
    if score01.shape != (H, W):
        score01 = cv2.resize(score01, (W, H), interpolation=cv2.INTER_LINEAR)
    cmap = plt.get_cmap("coolwarm")
    heat_rgb = (cmap(np.clip(score01, 0, 1))[:,:,:3]*255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)
    blend = (post_bgr.astype(np.float32)*(1-alpha) + heat_bgr.astype(np.float32)*alpha).astype(np.uint8)
    cv2.imwrite(out_png, blend)

# ---------------- Main Pipeline ----------------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    out_masks    = os.path.join(args.out, "masks")
    out_heat     = os.path.join(args.out, "heatmaps")
    out_heat_ov  = os.path.join(args.out, "heatmap_overlays")
    out_debug    = os.path.join(args.out, "debug")
    for d in [out_masks, out_heat, out_heat_ov, out_debug]:
        os.makedirs(d, exist_ok=True)

    # load and model
    pre_pil,  pre_t  = load_image_tensor(args.pre,  args.size)
    post_pil, post_t = load_image_tensor(args.post, args.size)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model = SiameseUNetResNet50(num_classes=1).to(device).eval()
    with torch.no_grad():
        prob_ = model(pre_t.to(device), post_t.to(device)).squeeze().cpu().numpy()  # [size,size] in [0,1]

    # model-based mask (for reference)
    model_mask = (prob_ > args.model_thresh).astype(np.uint8)*255

    # RGB diff (for sanity)
    pre_np  = pre_t.squeeze().cpu().numpy()
    post_np = post_t.squeeze().cpu().numpy()
    diff_gray = normalize01(np.mean(np.abs(post_np - pre_np), axis=0))

    # Build fused heat score at model resolution
    pre_rgb_256  = np.array(pre_pil.resize((args.size,args.size),  Image.BILINEAR))
    post_rgb_256 = np.array(post_pil.resize((args.size,args.size), Image.BILINEAR))

    L_pre  = to_lab_l(pre_rgb_256)
    L_post = to_lab_l(post_rgb_256)
    diff_L = np.abs(L_post - L_pre); diff_L = diff_L/(diff_L.max()+1e-6)

    edge_pre  = edge_mag_gray(pre_rgb_256)
    edge_post = edge_mag_gray(post_rgb_256)
    edge_diff = np.abs(edge_post - edge_pre); edge_diff = edge_diff/(edge_diff.max()+1e-6)

    water_256 = water_prior_rgb(post_rgb_256, blue_minus_green=args.blue_minus_green, max_r=args.max_r)

    # Ensure all components have the same resolution for fusion
    target_shape = (args.size, args.size)
    if prob_.shape != target_shape:
        prob_ = cv2.resize(prob_, target_shape, interpolation=cv2.INTER_LINEAR)
    if diff_L.shape != target_shape:
        diff_L = cv2.resize(diff_L, target_shape, interpolation=cv2.INTER_LINEAR)
    if edge_diff.shape != target_shape:
        edge_diff = cv2.resize(edge_diff, target_shape, interpolation=cv2.INTER_LINEAR)

    # fuse
    score = (args.w_L*diff_L + args.w_edge*edge_diff + args.w_prob*prob_)
    if args.use_water_gate:
        score = score * np.clip(water_256, 0, 1)

    # contrast & gamma
    score = contrast_stretch01(score, p_low=args.p_low, p_high=args.p_high)
    score = gamma_boost(score, gamma=args.gamma)

    # save debug maps (upsampled to original)
    post_bgr_full = cv2.cvtColor(np.array(post_pil), cv2.COLOR_RGB2BGR)
    H0, W0 = post_bgr_full.shape[:2]
    cv2.imwrite(os.path.join(out_debug, "change_prob.png"),
                (cv2.resize(prob_, (W0, H0))*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_debug, "water_prior.png"),
                (cv2.resize(water_256, (W0, H0))*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_debug, "diff_L.png"),
                (cv2.resize(diff_L, (W0, H0))*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_debug, "edge_diff.png"),
                (cv2.resize(edge_diff, (W0, H0))*255).astype(np.uint8))

    # binary flood from fused score (percentile threshold)
    thr = float(np.percentile(score, args.bin_percentile))
    bin_mask = (score > thr).astype(np.uint8)*255

    # save masks at original resolution
    mask_model_full = cv2.resize(model_mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
    mask_fused_full = cv2.resize(bin_mask,    (W0, H0), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_masks, "flood_mask_model.png"), mask_model_full)
    cv2.imwrite(os.path.join(out_masks, "flood_mask.png"),       mask_fused_full)

    # save heatmaps
    heat_full = cv2.resize(score, (W0, H0), interpolation=cv2.INTER_LINEAR)
    save_heatmap_standalone(heat_full, os.path.join(out_heat, "heatmap.png"))
    save_heatmap_overlay(post_bgr_full, heat_full, os.path.join(out_heat_ov, "heatmap_overlay.png"), alpha=args.overlay_alpha)

    print("[done]")
    print("Saved to:", os.path.abspath(args.out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre",  required=True, help="path to pre-disaster RGB image")
    ap.add_argument("--post", required=True, help="path to post-disaster RGB image")
    ap.add_argument("--out",  default="output", help="output directory")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")

    # model mask threshold
    ap.add_argument("--model_thresh", type=float, default=0.5)

    # fusion weights
    ap.add_argument("--w_L",   type=float, default=0.45, help="weight for L* diff")
    ap.add_argument("--w_edge",type=float, default=0.25, help="weight for edge diff")
    ap.add_argument("--w_prob",type=float, default=0.30, help="weight for model prob")

    # water gate + priors
    ap.add_argument("--use_water_gate", action="store_true")
    ap.add_argument("--blue_minus_green", type=int, default=6)
    ap.add_argument("--max_r",            type=int, default=205)

    # contrast & gamma
    ap.add_argument("--p_low",  type=float, default=2.0)
    ap.add_argument("--p_high", type=float, default=98.0)
    ap.add_argument("--gamma",  type=float, default=0.6)

    # binary from fused score
    ap.add_argument("--bin_percentile", type=float, default=80.0)

    # overlay
    ap.add_argument("--overlay_alpha", type=float, default=0.45)

    args = ap.parse_args()
    main(args)
