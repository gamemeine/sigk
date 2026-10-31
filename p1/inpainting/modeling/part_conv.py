import json
from copy import copy
from datetime import datetime
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset


class PartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.register_buffer("mask_kernel", torch.ones(1, in_channels, kernel_size, kernel_size))
        self.window_size = in_channels * kernel_size * kernel_size

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Here 1 means valid. Masks have the same channels as their feature maps.
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
        # Paper's U-Net adapted to 256x256 images.
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
        # Dataset convention: 1 means a hole, 0 means a known pixel.
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
        # Dataset mask: 1 means a hole. L1 terms are averaged over the entire image.
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

        # TV uses adjacent pixels both inside the one-pixel dilation of the holes.
        region = F.max_pool2d(mask, 3, stride=1, padding=1)
        horizontal = (composite[:, :, :, 1:] - composite[:, :, :, :-1]).abs()
        vertical = (composite[:, :, 1:, :] - composite[:, :, :-1, :]).abs()
        tv = (horizontal * region[:, :, :, 1:] * region[:, :, :, :-1]).sum()
        tv += (vertical * region[:, :, 1:, :] * region[:, :, :-1, :]).sum()
        tv /= target.numel()

        return valid + 6 * hole + 0.05 * perceptual + 120 * style + 0.1 * tv


def split_train_val(dataset: DIV2KDataset, val_split: float = 0.2, seed: int = 42):
    paths = list(dataset.image_paths)
    Random(seed).shuffle(paths)
    valid_count = int(len(paths) * val_split)
    if not 1 <= valid_count < len(paths):
        raise ValueError("val_split must leave at least one image in each split")

    train_ds, valid_ds = copy(dataset), copy(dataset)
    train_ds.image_paths = paths[valid_count:]
    valid_ds.image_paths = paths[:valid_count]
    valid_ds.n = None
    valid_ds.random_images = False
    train_ds.set_seed(seed)
    valid_ds.set_seed(seed)
    return train_ds, valid_ds


def train(
    model: PartConvModel,
    train_ds: DIV2KDataset,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: str = "cpu",
    seed: int = 42,
    val_size: float = 0.2,
    experiment_name: str = "pconv",
    resume: bool = False,
):
    train_ds, valid_ds = split_train_val(train_ds, val_size, seed)
    if not 1 <= batch_size <= len(train_ds):
        raise ValueError("batch_size must be at least 1 and no larger than the training dataset")

    torch.manual_seed(seed)
    rng = Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(json.dumps({
        "name": experiment_name,
        "class_path": "p1.inpainting.models.part_conv.PartConvModel",
        "init_args": {"channels": model.channels},
        "weights": "best.pt",
    }, indent=2))
    model.to(device)
    criterion = InpaintingLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    history = []
    best_loss = float("inf")
    start_epoch = 0
    state_path = output_dir / "training_state.pt"
    if resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        history = state["history"]
        best_loss = state["best_loss"]
        start_epoch = state["epoch"]
        rng.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng_state"])
        print(f"Resuming after epoch {start_epoch}, target: {epochs} epochs.")

    for epoch in range(start_epoch, epochs):
        train_ds.set_seed(rng.randrange(2**32))
        model.train()
        train_loss = 0.0
        train_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for cut, mask, target in progress:
            cut, mask, target = (tensor.to(device) for tensor in (cut, mask, target))
            optimizer.zero_grad()
            prediction = model(cut, mask)
            loss = criterion(prediction, mask, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * cut.shape[0]
            train_count += cut.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        valid_ds.set_seed(seed)
        valid_loss = 0.0
        hole_error = 0.0
        hole_values = 0
        with torch.no_grad():
            for cut, mask, target in valid_loader:
                cut, mask, target = (tensor.to(device) for tensor in (cut, mask, target))
                prediction = model(cut, mask)
                valid_loss += criterion(prediction, mask, target).item() * cut.shape[0]
                hole_error += ((prediction - target).abs() * mask).sum().item()
                hole_values += mask.sum().item() * target.shape[1]
        train_loss /= train_count
        valid_loss /= len(valid_ds)
        valid_hole_mae = hole_error / hole_values
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
            "valid_hole_mae": valid_hole_mae, "lr": current_lr,
        })
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
            f"hole_mae={valid_hole_mae:.4f}, lr={current_lr:g}"
        )

        torch.save(model.state_dict(), output_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
        (output_dir / "stats.json").write_text(json.dumps(history, indent=2))
        temporary_state = state_path.with_suffix(".pt.tmp")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "history": history,
            "random_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }, temporary_state)
        temporary_state.replace(state_path)

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("p1/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume from an existing output directory")
    parser.add_argument("--experiment-name", default="pconv")
    parser.add_argument("--cut-size", type=int, nargs="+", choices=[3, 32], default=[3, 32])
    parser.add_argument("--samples", type=int, default=5_000, help="Number of random training samples per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Total target number of epochs, including completed ones")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()
    if args.resume is not None:
        if args.output_dir is not None and args.output_dir.resolve() != args.resume.resolve():
            parser.error("--resume continues in its directory; omit --output-dir or use the same path")
        saved_args = json.loads((args.resume / "run_config.json").read_text())
        parser.set_defaults(**{name: saved_args[name] for name in vars(args) if name in saved_args})
        args = parser.parse_args()
        args.output_dir = args.resume
        model_config = json.loads((args.output_dir / "model_config.json").read_text())
        model_args = model_config["init_args"]
    else:
        if args.output_dir is None:
            args.output_dir = Path("p1/output") / "part-conv" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_args = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    torch.manual_seed(args.seed)
    train(
        PartConvModel(**model_args),
        DIV2KDataset(args.data_dir, "train", n=args.samples, seed=args.seed, cut_size=args.cut_size),
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        val_size=args.val_size,
        experiment_name=args.experiment_name,
        resume=args.resume is not None,
    )
