from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATA = Path("p1/data/raw/DIV2K")
DEFAULT_RUNS = Path("p1/output/denoising")
DEFAULT_WEIGHTS = Path("p1/checkpoints/denoising")
VARIANTS = ("drunet_lite", "rescross", "drunet_lite_x4", "rescross_x4")


def parser():
    p = argparse.ArgumentParser(prog="python -m p1.denoising.cli")
    p.add_argument("command", choices=("train-v2", "train-v3", "train-v4", "tune-baseline", "benchmark"))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    p.add_argument("--augment", choices=("none", "geometric", "full"), default="geometric")
    p.add_argument("--multi-gpu", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--retune", action="store_true")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    return p


def train_v2(a):
    from .legacy.train import run
    cfg = argparse.Namespace(
        data_dir=a.data_dir, output_dir=a.runs_dir / "v2",
        epochs=a.epochs or 30, batch_size=a.batch_size or 4,
        samples_per_epoch=a.samples_per_epoch or 2400,
        patch_size=a.patch_size, augment=a.augment, depth=17, features=64,
        lr=a.lr or 3e-4, device="auto", workers=2 if a.workers is None else a.workers,
        seed=42, resume=a.resume or (a.runs_dir / "v2" / "last.pt" if (a.runs_dir / "v2" / "last.pt").exists() else None),
        deterministic=False, allow_incomplete=a.smoke,
    )
    return run(cfg)


def train_v3(a):
    from .legacy.finetune_v3 import train
    initialized = a.checkpoint or a.weights_dir / "v2" / "best.pt"
    if a.resume is None and not initialized.exists():
        raise FileNotFoundError(f"V2 checkpoint not found: {initialized}")
    cfg = argparse.Namespace(
        data_dir=a.data_dir, output_dir=a.runs_dir / "v3",
        epochs=a.epochs or 18, batch_size=a.batch_size or 16,
        samples_per_epoch=a.samples_per_epoch or 4800,
        patch_size=a.patch_size, augment=a.augment, depth=17, features=64,
        lr=a.lr or 3e-5, lr_min=1e-6,
        device="auto", workers=2 if a.workers is None else a.workers,
        seed=42, init_checkpoint=initialized,
        resume=a.resume or (a.runs_dir / "v3" / "last.pt" if (a.runs_dir / "v3" / "last.pt").exists() else None),
        ssim_weight=1.0, ema_decay=0.999, psnr_guard_db=0.15,
        amp=not a.no_amp, multi_gpu=a.multi_gpu, allow_incomplete=a.smoke,
    )
    return train(cfg)


def train_v4(a):
    from .v4.training import run_train
    settings = dict(seed=42, epochs=a.epochs or 14, batch=a.batch_size or 32,
                    samples_per_epoch=a.samples_per_epoch or 4800,
                    patch_size=a.patch_size, lr=a.lr or 2e-4, lr_min=1e-6,
                    ssim_weight=1.0, ema_decay=0.999,
                    workers=4 if a.workers is None else a.workers,
                    amp=not a.no_amp, multi_gpu=a.multi_gpu,
                    drunet_width=32, rescross_width=64)
    for variant in (VARIANTS if a.variant == "all" else (a.variant,)):
        run_train(variant, a.data_dir, a.runs_dir / variant, settings,
                  augmentation_x4=variant.endswith("_x4"), smoke=a.smoke)


def benchmark(a):
    from .v4.benchmark import benchmark as run_benchmark
    from .v4.benchmark import tune_bilateral
    baseline = a.weights_dir / "bilateral_val_tuned.json"
    if a.retune or not baseline.exists():
        baseline = a.runs_dir / "bilateral_val_tuned.json"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        tune_bilateral(a.data_dir, baseline, smoke=a.smoke)
    model_files = {
        "DnCNN V2": a.weights_dir / "v2" / "best.pt",
        "DnCNN V3": a.weights_dir / "v3" / "best.pt",
        "DRUNet-Lite": a.weights_dir / "drunet_lite" / "best.pt",
        "ResCrossAttention": a.weights_dir / "rescross" / "best.pt",
        "DRUNet-Lite ×4": a.weights_dir / "drunet_lite_x4" / "best.pt",
        "ResCrossAttention ×4": a.weights_dir / "rescross_x4" / "best.pt",
    }
    for label, path in tuple(model_files.items()):
        local = a.runs_dir / ({
            "DnCNN V2": "v2", "DnCNN V3": "v3",
            "DRUNet-Lite": "drunet_lite", "ResCrossAttention": "rescross",
            "DRUNet-Lite ×4": "drunet_lite_x4", "ResCrossAttention ×4": "rescross_x4",
        }[label]) / "best.pt"
        if local.exists():
            model_files[label] = local
    return run_benchmark(
        a.data_dir, a.runs_dir / "benchmark", model_files, baseline,
        seed=123, patch_size=a.patch_size, lpips=not a.no_lpips,
        smoke=a.smoke, max_images=a.max_images,
    )


def main():
    a = parser().parse_args()
    if a.smoke and a.max_images is None:
        a.max_images = 2
    if a.command == "train-v2":
        train_v2(a)
    elif a.command == "train-v3":
        train_v3(a)
    elif a.command == "train-v4":
        train_v4(a)
    elif a.command == "benchmark":
        benchmark(a)
    elif a.command == "tune-baseline":
        from .v4.benchmark import tune_bilateral
        output = a.runs_dir / "bilateral_val_tuned.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        tune_bilateral(a.data_dir, output, smoke=a.smoke)


if __name__ == "__main__":
    main()
