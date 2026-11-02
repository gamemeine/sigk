import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv2(F.silu(self.conv1(x)))
        return F.silu(x + h)


class ResidualUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), blocks_per_level=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, channels[0], 3, padding=1), nn.SiLU())
        self.encoder = nn.ModuleList(
            nn.Sequential(*(ResBlock(c) for _ in range(blocks_per_level))) for c in channels
        )
        self.down = nn.ModuleList(
            nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(channels, channels[1:])
        )
        self.decoder = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(a + b, b, 3, padding=1), nn.SiLU(),
                *(ResBlock(b) for _ in range(blocks_per_level)),
            )
            for a, b in zip(reversed(channels[1:]), reversed(channels[:-1]))
        )
        self.head = nn.Sequential(nn.Conv2d(channels[0], 3, 3, padding=1), nn.Sigmoid())
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        x = self.stem(inputs)
        skips = []
        for level, blocks in enumerate(self.encoder):
            x = blocks(x)
            if level < len(self.down):
                skips.append(x)
                x = self.down[level](x)
        for blocks, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blocks(torch.cat([x, skip], dim=1))
        return self.head(x)
