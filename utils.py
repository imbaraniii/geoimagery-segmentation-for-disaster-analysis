import os
from PIL import Image
from torchvision import transforms

def make_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def load_tensor_rgb(path, size):
    pil = Image.open(path).convert("RGB")
    return pil, make_tf(size)(pil).unsqueeze(0)  # [1,3,H,W]

def find_pairs(pre_dir, post_dir, pre_suffix="_pre", post_suffix="_post"):
    """Pairs by matching filenames or replacing suffix."""
    pairs = []
    for name in sorted(os.listdir(pre_dir)):
        p = os.path.join(pre_dir, name)
        if not os.path.isfile(p): continue
        candidates = [name.replace(pre_suffix, post_suffix), name]
        for c in candidates:
            q = os.path.join(post_dir, c)
            if os.path.exists(q):
                pairs.append((p, q, os.path.splitext(name)[0]))
                break
    return pairs
