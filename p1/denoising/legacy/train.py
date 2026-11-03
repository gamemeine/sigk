"""Trening DnCNN: MSE na szumie znormalizowanym / sigma; walidacja PSNR, checkpoint + CSV."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DIV2KDenoising, SIGMAS, fixed_noise
from .model import DnCNN


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Nie wykryto CUDA. Uzyj --device cpu albo sprawdz sterowniki/PyTorch.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return -10 * torch.log10(mse)


@torch.no_grad()
def validate(model: DnCNN, dataset: DIV2KDenoising, device: torch.device, seed: int) -> dict:
    model.eval()
    totals = {sigma: [] for sigma in SIGMAS}
    for i in range(len(dataset)):
        _, clean, _ = dataset[i]
        for sigma in SIGMAS:
            noisy = fixed_noise(clean, sigma, seed, i).unsqueeze(0).to(device)
            target = clean.unsqueeze(0).to(device)
            prediction = model(noisy, sigma).clamp(0, 1)
            totals[sigma].append(psnr_per_image(prediction, target).item())
    return {f"val_psnr_sigma_{sigma:.2f}": float(np.mean(totals[sigma])) for sigma in SIGMAS}


def run(args: argparse.Namespace):
    if args.epochs < 1 or args.batch_size < 1 or args.samples_per_epoch < 1:
        raise ValueError("epochs, batch_size i samples_per_epoch musza byc > 0")
    if args.patch_size % 8:
        raise ValueError("patch_size powinien byc wielokrotnoscia 8 (standard 256)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False

    train_ds = DIV2KDenoising(args.data_dir, "train", size=args.patch_size, augment=args.augment,
                             samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    val_ds = DIV2KDenoising(args.data_dir, "val", size=args.patch_size, augment="none", seed=args.seed)
    if not args.allow_incomplete and (len(train_ds.paths) != 800 or len(val_ds.paths) != 50):
        raise RuntimeError(f"DIV2K niekompletny: train={len(train_ds.paths)}/800, val={len(val_ds.paths)}/50. "
                           "Do lokalnych testow uzyj --allow-incomplete.")
    device = resolve_device(args.device)
    model = DnCNN(depth=args.depth, features=args.features, conditioned=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_psnr = 0, float("-inf")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = checkpoint["model_config"]
        if (expected["depth"], expected["features"]) != (args.depth, args.features):
            raise ValueError("Przy --resume podaj te same --depth i --features jak w checkpoincie")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_psnr = checkpoint["epoch"], checkpoint["best_psnr"]
        if start_epoch >= args.epochs:
            raise ValueError("--epochs musi byc wieksze od liczby epok w checkpoincie")
        print(f"Resume: zaczynam od epoki {start_epoch+1}, najlepszy PSNR {best_psnr:.4f}")

    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    config["resume"] = str(config["resume"]) if config["resume"] else None
    config["n_training_images"] = len(train_ds.paths)
    config["n_validation_images"] = len(val_ds.paths)
    config["train_ids"] = [p.stem for p in train_ds.paths]
    config["val_ids"] = [p.stem for p in val_ds.paths]
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = out / "history.csv"
    if not args.resume and log_path.exists():
        raise FileExistsError(f"Historia juz istnieje: {log_path}. Wybierz inny --output-dir.")
    columns = ["epoch", "train_mse", "train_normalized_mse", "lr", "val_psnr_sigma_0.01", "val_psnr_sigma_0.03", "val_psnr_mean"]
    with log_path.open("a", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=columns)
        if log_path.stat().st_size == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            train_ds.set_epoch(epoch)
            gen = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=device.type == "cuda", generator=gen)
            model.train()
            running_loss, running_norm_loss, seen = 0.0, 0.0, 0
            for noisy, clean, sigma in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
                noisy, clean, sigma = noisy.to(device), clean.to(device), sigma.to(device)
                optimizer.zero_grad(set_to_none=True)
                result = model(noisy, sigma)
                loss = (((result - clean) / sigma.view(-1, 1, 1, 1)) ** 2).mean()  # uczymy residual / sigma
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += nn.functional.mse_loss(result.detach(), clean).item() * len(noisy)
                running_norm_loss += loss.item() * len(noisy)
                seen += len(noisy)
            scores = validate(model, val_ds, device, args.seed + 1_000_000)
            score = float(np.mean(list(scores.values())))
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            row = dict(epoch=epoch+1, train_mse=running_loss/seen, train_normalized_mse=running_norm_loss/seen, lr=current_lr,
                       val_psnr_mean=score, **scores)
            writer.writerow(row)
            log.flush()
            is_best = score > best_psnr
            if is_best:
                best_psnr = score
            state = dict(
                epoch=epoch+1, best_psnr=best_psnr, model_config={"depth": args.depth, "features": args.features,
                                                                 "conditioned": True},
                state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                train_config=config,
            )
            torch.save(state, out / "last.pt")
            if is_best:
                torch.save(state, out / "best.pt")
            print(f"Epoka {epoch+1:>3}: MSE={row['train_mse']:.6f}, val PSNR "
                  f"σ=.01 {scores['val_psnr_sigma_0.01']:.3f} / "
                  f"σ=.03 {scores['val_psnr_sigma_0.03']:.3f}, mean={score:.3f} dB" + (" *BEST*" if is_best else ""))
    print(f"Gotowe: {out/'best.pt'} | historia: {log_path}")


def parser():
    p = argparse.ArgumentParser(description="Trening DnCNN na DIV2K (SIGK 2.2)")
    p.add_argument("--data-dir", type=Path, default=Path("data/DIV2K"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/dncnn_geometric"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=3200)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--augment", choices=["none", "geometric", "full"], default="geometric")
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=0, help="Windows: domyslnie 0 (mozna zwiekszyc)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true", help="TYLKO smoke-test na malym zbiorze")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
