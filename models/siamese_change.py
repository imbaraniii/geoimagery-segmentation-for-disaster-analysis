import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- ResNet50 backbone with version fallback ----
def _get_resnet50():
    from torchvision.models import resnet50
    try:
        from torchvision.models import ResNet50_Weights
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        return resnet50(pretrained=True)

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = _get_resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.layer0(x)  # [B,  64, H/4,  W/4]
        x1 = self.layer1(x0) # [B, 256, H/4,  W/4]
        x2 = self.layer2(x1) # [B, 512, H/8,  W/8]
        x3 = self.layer3(x2) # [B,1024, H/16, W/16]
        x4 = self.layer4(x3) # [B,2048, H/32, W/32]
        return x0, x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1   = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.b1   = nn.BatchNorm2d(out_ch)
        self.c2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.b2   = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x    = torch.cat([x, skip], dim=1)
        x    = F.relu(self.b1(self.c1(x)))
        x    = F.relu(self.b2(self.c2(x)))
        return x

class SiameseUNetResNet50(nn.Module):
    """Siamese shared encoder; deep abs-diff; U-Net style decoder to 1xHxW prob."""
    def __init__(self, out_ch=1):
        super().__init__()
        self.enc  = ResNet50Encoder()
        self.dec4 = DecoderBlock(2048, 1024*2, 512)
        self.dec3 = DecoderBlock(512,  512*2,  256)
        self.dec2 = DecoderBlock(256,  256*2,  128)
        self.dec1 = DecoderBlock(128,   64*2,   64)
        self.out  = nn.Conv2d(64, out_ch, 1)

    def forward(self, pre, post):
        p0,p1,p2,p3,p4 = self.enc(pre)
        q0,q1,q2,q3,q4 = self.enc(post)
        x = torch.abs(q4 - p4)
        x = self.dec4(x, torch.cat([p3,q3], dim=1))
        x = self.dec3(x, torch.cat([p2,q2], dim=1))
        x = self.dec2(x, torch.cat([p1,q1], dim=1))
        x = self.dec1(x, torch.cat([p0,q0], dim=1))
        return torch.sigmoid(self.out(x))  # [B,1,H,W] in [0,1]
