import torch
from torch import nn
from torch.nn import functional as F


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, 3, stride=stride, padding=1)
        self.register_buffer("mask_kernel", torch.ones(1, in_channels, 3, 3), persistent=False)

    def forward(self, x, mask):
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            count = F.conv2d(mask.float(), self.mask_kernel.float(), stride=self.stride, padding=self.padding)
            updated_mask = (count > 0).float()
            ratio = self.in_channels * 9 / count.clamp_min(1) * updated_mask

        raw = super().forward(x * mask)
        bias = self.bias.view(1, -1, 1, 1)
        output = ((raw.float() - bias) * ratio + bias) * updated_mask
        return output.to(raw.dtype), updated_mask.expand(-1, self.out_channels, -1, -1)


class PartialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = PartialConv2d(channels, channels)
        self.conv2 = PartialConv2d(channels, channels)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(F.silu(x), mask)
        return F.silu(x), mask


class PartialUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), blocks_per_level=2):
        super().__init__()
        self.stem = PartialConv2d(3, channels[0])
        self.encoder = nn.ModuleList(nn.ModuleList(PartialBlock(c) for _ in range(blocks_per_level)) for c in channels)
        self.down = nn.ModuleList(PartialConv2d(a, b, stride=2) for a, b in zip(channels, channels[1:]))
        self.reduce = nn.ModuleList(
            PartialConv2d(a + b, b) for a, b in zip(reversed(channels[1:]), reversed(channels[:-1]))
        )
        self.decoder = nn.ModuleList(
            nn.ModuleList(PartialBlock(c) for _ in range(blocks_per_level)) for c in reversed(channels[:-1])
        )
        self.head = PartialConv2d(channels[0], 3)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        x = inputs[:, :3]
        mask = (1 - inputs[:, 3:4]).expand_as(x)
        x, mask = self.stem(x, mask)
        x = F.silu(x)
        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                x, mask = block(x, mask)
            if level < len(self.down):
                skips.append((x, mask))
                x, mask = self.down[level](x, mask)

        for reduce, blocks, (skip, skip_mask) in zip(self.reduce, self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=skip.shape[-2:], mode="nearest")
            x, mask = reduce(torch.cat([x, skip], dim=1), torch.cat([mask, skip_mask], dim=1))
            x = F.silu(x)
            for block in blocks:
                x, mask = block(x, mask)
        x, _ = self.head(x, mask)
        return torch.sigmoid(x)
