import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Optional deps for visualization
try:
    import cv2
except Exception:
    cv2 = None

try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    _HAS_CAM = True
except Exception:
    _HAS_CAM = False


# ---------------- Basic Blocks ----------------

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """
    Paper-style channel attention block:

      AdaptiveAvgPool2d(1) →
      1×1 conv (C -> C/r) + SELU →
      1×1 conv (C/r -> C) + Sigmoid →
      scale original feature maps

    This matches the legend:
      - yellow: adaptive avg pooling
      - green: SELU
      - purple: Sigmoid
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SELU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))  # (B,C,1,1)
        return x * w


class MultiScaleBlock(nn.Module):
    """
    One AMCNN multi-scale block:

      branch1: 3×3 conv
      branch2: 5×5 conv
      branch3: 1×1 conv

    All branches output 'out_ch' channels and use the same stride.
    Outputs are SUM'ed then scaled (divide by 3) – as in the diagram:
      three branches → Sum → ×Scale.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.branch_3x3 = ConvBlock(in_ch, out_ch, k=3, s=stride, p=1)
        self.branch_5x5 = ConvBlock(in_ch, out_ch, k=5, s=stride, p=2)
        self.branch_1x1 = ConvBlock(in_ch, out_ch, k=1, s=stride, p=0)

    def forward(self, x):
        b1 = self.branch_3x3(x)
        b2 = self.branch_5x5(x)
        b3 = self.branch_1x1(x)
        # Sum + scale (average) like "Sum ×Scale" in the figure
        y = (b1 + b2 + b3) / 3.0
        return y


# ---------------- STN (alignment module) ----------------

class STNLocalization(nn.Module):
    """
    Predicts [sx_raw, sy_raw, tx, ty, θ]
    Scale is constrained around 1.0: sx = 1 + scale_factor * sx_raw
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3   = nn.BatchNorm2d(64)

        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, 5)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)  # (B,5)


class STNAligner(nn.Module):
    """
    STN with scale + rotation + translation.
    """
    def __init__(self, in_channels: int = 3, scale_factor: float = 0.3):
        super().__init__()
        self.loc = STNLocalization(in_channels)
        self.scale_factor = scale_factor  # used by dump script

    def forward(self, x):
        B, C, H, W = x.shape

        params = self.loc(x)  # (B,5)
        sx_raw, sy_raw, tx, ty, theta = params.split(1, dim=1)

        sx = 1.0 + self.scale_factor * sx_raw
        sy = 1.0 + self.scale_factor * sy_raw

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        A = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        A[:, 0, 0] = sx[:, 0] * cos_t[:, 0]
        A[:, 0, 1] = -sx[:, 0] * sin_t[:, 0]
        A[:, 1, 0] = sy[:, 0] * sin_t[:, 0]
        A[:, 1, 1] = sy[:, 0] * cos_t[:, 0]
        A[:, 0, 2] = tx[:, 0]
        A[:, 1, 2] = ty[:, 0]

        grid = F.affine_grid(A, x.size(), align_corners=False)
        x_aligned = F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )
        return x_aligned


# ---------------- AMCNN encoder (single branch) ----------------

class STAMCNNEncoder(nn.Module):
    """
    True AMCNN-style encoder matching the diagram:

      Stage 1: Multi-scale (4 channels) + attention
      Stage 2: Multi-scale (8 channels) + attention
      Then:   3×3 conv → 16 ch
              3×3 conv → 32 ch
              Global avg pool → FC(feature_dim=64 by default)

    For Grad-CAM we hook into the last 3×3 conv (32 channels).
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        # 👉 This is the extra CNN before the first parallel block
        self.stem = ConvBlock(in_channels, 4, k=3, s=1)   # 3→4
    
        # --- Stage 1: multi-scale, 4 channels ---
        self.ms1 = MultiScaleBlock(in_ch=4, out_ch=4)   # 3×3,3×3,5×5 branches, sum→4
        self.attn1 = ChannelAttention(4, reduction=2)
        self.conv_to8 = ConvBlock(4, 8, k=3, s=1)
        # --- Stage 2: multi-scale, 8 channels ---
        self.ms2   = MultiScaleBlock(8, out_ch=16, stride=1)
        self.attn2 = ChannelAttention(16, reduction=2)

        # --- Single-scale convs to 16 and 32 channels ---
        self.conv_to32 = ConvBlock(16, 32, k=3, s=1)
        self.conv_to64 = ConvBlock(32, 64, k=3, s=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, feature_dim)

        # Grad-CAM target layer = last conv
        self.target_layer = self.conv_to64.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.stem(x)
        # Stage 1
        x = self.ms1(x)
        x = self.attn1(x)
        x = self.conv_to8(x)
        # Stage 2
        x = self.ms2(x)
        x = self.attn2(x)

        # Final convs
        x = self.conv_to32(x)
        x = self.conv_to64(x)

        # Global pooling + FC
        g = self.pool(x).flatten(1)   # (B,32)
        z = self.fc(g)                # (B, feature_dim)
        z = F.normalize(z, p=2, dim=1)  # for cosine similarity
        return z


# ---------------- Siamese STN + AMCNN ----------------

class SiameseSTN_AMCNN(nn.Module):
    """
    branch 1: standard pose  I_s → encoder
    branch 2: learner pose   I_u → (optional STN) → encoder
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 64, use_stn: bool = True):
        super().__init__()
        self.use_stn = use_stn
        self.stn     = STNAligner(in_channels) if use_stn else None
        self.encoder = STAMCNNEncoder(in_channels, feature_dim)

    def forward(self, x_std: torch.Tensor, x_learner: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_std = self.encoder(x_std)

        if self.use_stn and self.stn is not None:
            x_learner = self.stn(x_learner)
        z_learner = self.encoder(x_learner)

        return z_std, z_learner

    def encode_standard(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_learner(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_stn and self.stn is not None:
            x = self.stn(x)
        return self.encoder(x)



