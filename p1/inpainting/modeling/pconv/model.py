import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16


class PartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.register_buffer("mask_kernel", torch.ones(1, in_channels, kernel_size, kernel_size))
        self.window_size = in_channels * kernel_size * kernel_size

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = F.conv2d(mask, self.mask_kernel, stride=self.conv.stride, padding=self.conv.padding)
        updated_mask = (count > 0).to(image.dtype)
        ratio = self.window_size / count.clamp_min(1)

        output = self.conv(image * mask)
        bias = self.conv.bias.view(1, -1, 1, 1)
        output = ((output - bias) * ratio + bias) * updated_mask
        return output, updated_mask.expand(-1, output.shape[1], -1, -1)


class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, batch_norm, activation):
        super().__init__()
        self.partial_conv = PartialConv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.activation = activation

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.partial_conv(image, mask)
        return self.activation(self.norm(image)), mask


class PartConvModel(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256, 256, 256, 256)):
        super().__init__()
        self.channels = list(channels)
        depth = len(self.channels)
        channels = [3, *self.channels]
        kernels = [7, 5, 5] + [3] * (depth - 3)
        self.encoder = nn.ModuleList([
            PartialConvBlock(channels[i], channels[i + 1], kernels[i], 2, i > 0, nn.ReLU())
            for i in range(depth)
        ])
        self.decoder = nn.ModuleList([
            PartialConvBlock(
                channels[i + 1] + channels[i], channels[i], 3, 1,
                i > 0, nn.LeakyReLU(0.2) if i > 0 else nn.Identity(),
            )
            for i in reversed(range(depth))
        ])

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = (1 - mask).expand_as(image)
        skips = [(image, mask)]
        for block in self.encoder:
            image, mask = block(image, mask)
            skips.append((image, mask))

        for block, (skip_image, skip_mask) in zip(self.decoder, reversed(skips[:-1])):
            image = F.interpolate(image, size=skip_image.shape[-2:], mode="nearest")
            mask = F.interpolate(mask, size=skip_mask.shape[-2:], mode="nearest")
            image = torch.cat((image, skip_image), dim=1)
            mask = torch.cat((mask, skip_mask), dim=1)
            image, mask = block(image, mask)

        return image

    @torch.no_grad()
    def predict(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        prediction = self(image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
        prediction = prediction.squeeze(0).cpu().clamp(0, 1)
        return image * (1 - mask) + prediction * mask


def _gram(features: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = features.shape
    features = features.reshape(batch, channels, height * width)
    return features @ features.transpose(1, 2) / (channels * height * width)


class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.ModuleList([features[:5], features[5:10], features[10:17]])
        self.vgg.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = (image - self.mean) / self.std
        features = []
        for block in self.vgg:
            image = block(image)
            features.append(image)
        return features

    def forward(self, prediction: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (prediction - target).abs()
        valid = ((1 - mask) * error).mean()
        hole = (mask * error).mean()
        composite = target * (1 - mask) + prediction * mask

        output_features = self._features(prediction)
        composite_features = self._features(composite)
        target_features = self._features(target)
        perceptual = 0
        style = 0
        for output, comp, gt in zip(output_features, composite_features, target_features):
            perceptual += F.l1_loss(output, gt) + F.l1_loss(comp, gt)
            style += F.l1_loss(_gram(output), _gram(gt)) + F.l1_loss(_gram(comp), _gram(gt))

        region = F.max_pool2d(mask, 3, stride=1, padding=1)
        horizontal = (composite[:, :, :, 1:] - composite[:, :, :, :-1]).abs()
        vertical = (composite[:, :, 1:, :] - composite[:, :, :-1, :]).abs()
        tv = (horizontal * region[:, :, :, 1:] * region[:, :, :, :-1]).sum()
        tv += (vertical * region[:, :, 1:, :] * region[:, :, :-1, :]).sum()
        tv /= target.numel()

        return valid + 6 * hole + 0.05 * perceptual + 120 * style + 0.1 * tv
