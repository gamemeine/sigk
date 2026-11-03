import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16


def masked_mean(values, mask):
    return (values * mask).sum() / (mask.sum() * values.shape[1]).clamp_min(1)


def gram_matrix(features):
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.float().flatten(2)
        return features.bmm(features.transpose(1, 2)) / (features.shape[1] * features.shape[2])


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:17].eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        features = []
        for index, layer in enumerate(self.layers):
            image = layer(image)
            if index in (4, 9, 16):
                features.append(image)
        return features


class InpaintingLoss(nn.Module):
    def __init__(
        self, l1_weight=1.0, mse_weight=0.0, valid_weight=0.0, gradient_weight=0.0,
        perceptual_weight=0.0, style_weight=0.0, tv_weight=0.0, perceptual_crop_size=64,
    ):
        super().__init__()
        self.weights = {
            "l1": l1_weight, "mse": mse_weight, "valid": valid_weight, "gradient": gradient_weight,
            "perceptual": perceptual_weight, "style": style_weight, "tv": tv_weight,
        }
        self.perceptual_crop_size = perceptual_crop_size
        self.vgg = VGGFeatures() if perceptual_weight or style_weight else None

    def _hole_crops(self, composite, target, mask):
        height, width = target.shape[-2:]
        size = min(self.perceptual_crop_size, height, width)
        crops = []
        for image, hole in zip(torch.cat([composite, target], dim=1), mask):
            positions = hole[0].nonzero()
            top = int(((positions[:, 0].min() + positions[:, 0].max() + 1 - size) // 2).clamp(0, height - size))
            left = int(((positions[:, 1].min() + positions[:, 1].max() + 1 - size) // 2).clamp(0, width - size))
            crops.append(image[:, top:top + size, left:left + size])
        crops = torch.stack(crops)
        return crops[:, :3], crops[:, 3:]

    def forward(self, prediction, target, mask):
        prediction = prediction.float()
        error = prediction - target
        weights = self.weights
        loss = prediction.sum() * 0
        if weights["l1"]:
            loss = loss + weights["l1"] * masked_mean(error.abs(), mask)
        if weights["mse"]:
            loss = loss + weights["mse"] * masked_mean(error.square(), mask)
        if weights["valid"]:
            loss = loss + weights["valid"] * masked_mean(error.abs(), 1 - mask)

        composite = target * (1 - mask) + prediction * mask
        if weights["gradient"] or weights["tv"]:
            region = F.max_pool2d(mask, 3, stride=1, padding=1)
            for dim in (-2, -1):
                size = mask.shape[dim] - 1
                difference = composite.diff(dim=dim)
                if weights["gradient"]:
                    edges = mask.narrow(dim, 0, size).maximum(mask.narrow(dim, 1, size))
                    gradient_error = (difference - target.diff(dim=dim)).abs()
                    loss = loss + weights["gradient"] * masked_mean(gradient_error, edges) / 2
                if weights["tv"]:
                    edges = region.narrow(dim, 0, size) * region.narrow(dim, 1, size)
                    loss = loss + weights["tv"] * masked_mean(difference.abs(), edges) / 2

        if self.vgg is not None:
            composite_crop, target_crop = self._hole_crops(composite, target, mask)
            predicted_features = self.vgg(composite_crop)
            with torch.no_grad():
                target_features = self.vgg(target_crop)
            for predicted, reference in zip(predicted_features, target_features):
                if weights["perceptual"]:
                    loss = loss + weights["perceptual"] * F.l1_loss(predicted, reference) / 3
                if weights["style"]:
                    loss = loss + weights["style"] * F.l1_loss(gram_matrix(predicted), gram_matrix(reference)) / 3
        return loss


class PartConvLoss(nn.Module):
    def __init__(self, hole_weight=6.0, valid_weight=1.0, perceptual_weight=5e-2, style_weight=120.0, tv_weight=1e-1):
        super().__init__()
        self.weights = {
            "hole": hole_weight, "valid": valid_weight, "perceptual": perceptual_weight,
            "style": style_weight, "tv": tv_weight,
        }
        self.vgg = VGGFeatures() if perceptual_weight or style_weight else None

    def forward(self, prediction, target, mask):
        prediction = prediction.float()
        error = (prediction - target).abs()
        weights = self.weights
        loss = weights["hole"] * (error * mask).mean() + weights["valid"] * (error * (1 - mask)).mean()
        composite = target * (1 - mask) + prediction * mask

        if self.vgg is not None:
            predicted_features = self.vgg(prediction)
            composite_features = self.vgg(composite)
            with torch.no_grad():
                target_features = self.vgg(target)
            for predicted, composited, reference in zip(predicted_features, composite_features, target_features):
                if weights["perceptual"]:
                    loss = loss + weights["perceptual"] * (
                        F.l1_loss(predicted, reference) + F.l1_loss(composited, reference)
                    )
                if weights["style"]:
                    reference_gram = gram_matrix(reference)
                    loss = loss + weights["style"] * (
                        F.l1_loss(gram_matrix(predicted), reference_gram)
                        + F.l1_loss(gram_matrix(composited), reference_gram)
                    )

        if weights["tv"]:
            region = F.max_pool2d(mask, 3, stride=1, padding=1)
            for dim in (-2, -1):
                size = mask.shape[dim] - 1
                edges = region.narrow(dim, 0, size) * region.narrow(dim, 1, size)
                loss = loss + weights["tv"] * (composite.diff(dim=dim).abs() * edges).sum() / composite.numel()
        return loss
