"""SIGK 2.2: DRUNet-Lite and residual CNN with true, resolution-limited cross-attention.
Outputs are denoised RGB, using V2-compatible normalized-noise prediction y - sigma * noise_hat.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .legacy import DnCNN


def prepare_input(noisy, sigma):
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=noisy.device, dtype=noisy.dtype)
    sigma = sigma.to(device=noisy.device, dtype=noisy.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(noisy.size(0))
    if sigma.shape != (noisy.size(0),):
        raise ValueError('Sigma must be scalar or B-sized vector')
    sigmamap = (sigma / 0.03)[:, None, None, None].expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
    return torch.cat([noisy, sigmamap], dim=1), sigma


def norm(ch):
    for groups in (8, 4, 2, 1):
        if ch % groups == 0:
            return nn.GroupNorm(groups, ch)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), norm(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), norm(ch))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.net(x)


class DRUNetLite(nn.Module):
    """Three-level U-Net + residual blocks; unlike original DRUNet this is a lightweight variant."""
    def __init__(self, width=32):
        super().__init__()
        c1, c2, c3 = width, width*2, width*3
        self.stem = nn.Conv2d(4, c1, 3, padding=1)
        self.e1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2))
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3), ResidualBlock(c3))
        self.up2 = nn.Conv2d(c3+c2, c2, 3, padding=1)
        self.d2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2))
        self.up1 = nn.Conv2d(c2+c1, c1, 3, padding=1)
        self.d1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))
        self.head = nn.Conv2d(c1, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, sigma):
        inp, sigma = prepare_input(noisy, sigma)
        x1 = self.e1(self.stem(inp))
        x2 = self.e2(self.down1(x1))
        x3 = self.mid(self.down2(x2))
        x = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.d2(self.up2(torch.cat((x, x2), dim=1)))
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.d1(self.up1(torch.cat((x, x1), dim=1)))
        return noisy - sigma[:, None, None, None] * self.head(x)


class CrossContext(nn.Module):
    """Q from learned local features; K/V from a DIFFERENT global context branch.

    At 256x256, Q=16x16=256 tokens; K/V=8x8=64 tokens, four heads.
    This is actual cross-attention rather than a channel attention / SE block.
    """
    def __init__(self, width=64, heads=4):
        super().__init__()
        assert width % heads == 0
        self.local_projection = nn.Conv2d(width, width, 1)
        self.context = nn.Sequential(
            nn.Conv2d(4, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.post = nn.Conv2d(width, width, 1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, local, original_in):
        b, c, h, w = local.shape
        q_map = F.adaptive_avg_pool2d(self.local_projection(local), (16,16))
        kv_map = F.adaptive_avg_pool2d(self.context(original_in), (8,8))
        q = q_map.flatten(2).transpose(1,2)
        kv = kv_map.flatten(2).transpose(1,2)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        out = attended.transpose(1,2).reshape(b,c,16,16)
        out = F.interpolate(self.post(out), size=(h,w), mode='bilinear', align_corners=False)
        return local + self.gate * out


class ResCrossDnCNN(nn.Module):
    """Residual denoiser with one efficient local-to-global cross-attention bridge."""
    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Conv2d(4, width, 3, padding=1)
        self.before = nn.Sequential(*[ResidualBlock(width) for _ in range(4)])
        self.cross = CrossContext(width, heads=4)
        self.after = nn.Sequential(*[ResidualBlock(width) for _ in range(4)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, sigma):
        inp, sigma = prepare_input(noisy, sigma)
        x = self.before(self.stem(inp))
        x = self.cross(x, inp)
        x = self.after(x)
        return noisy - sigma[:, None, None, None] * self.head(x)


def make_model(name, config=None):
    config = config or {}
    if name in ('v2', 'v3', 'dncnn'):
        return DnCNN(**(config or {'depth': 17, 'features':64, 'conditioned':True}))
    if name == 'drunet_lite':
        return DRUNetLite(width=config.get('width',32))
    if name == 'rescross':
        return ResCrossDnCNN(width=config.get('width',64))
    raise ValueError('Unrecognized model: ' + str(name))
