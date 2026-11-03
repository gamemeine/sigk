"""Ewaluacja na rozdzielonym DIV2K-valid: PSNR/SSIM/LPIPS oraz obrazy i CSV."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from .baseline import bilateral
from .data import DIV2KDenoising, SIGMAS, fixed_noise
from .model import DnCNN
from .train import resolve_device


def build_lpips(device: torch.device):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        return metric
    except Exception as exc:
        raise RuntimeError(
            "LPIPS nie jest dostepny. Zainstaluj 'pip install -e .[lpips]' "
            "i zezwol na pierwsze pobranie wag AlexNet. "
            "Do smoke-testu mozesz uzyc --skip-lpips (nie do raportu na zajecia)."
        ) from exc


def tensor_to_hwc(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def metrics(pred: torch.Tensor, clean: torch.Tensor, lpips_metric, device):
    a = tensor_to_hwc(pred.clamp(0, 1))
    b = tensor_to_hwc(clean)
    result = {
        "PSNR": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
        "SSIM": float(structural_similarity(b, a, channel_axis=-1, data_range=1.0)),
        "LPIPS": None,
    }
    if lpips_metric is not None:
        with torch.no_grad():
            result["LPIPS"] = float(lpips_metric(pred.clamp(0, 1).unsqueeze(0).to(device),
                                                 clean.unsqueeze(0).to(device)).item())
    return result


def save_visuals(rows: list, path: Path):
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.5 * len(rows)), squeeze=False)
    for row, (title, images) in enumerate(rows):
        axes[row, 0].set_ylabel(title, fontsize=10)
        for col, (name, img) in enumerate(zip(("Zaszumiony", "Bilateral", "DnCNN", "Oryginal"), images)):
            axes[row, col].imshow(tensor_to_hwc(img.clamp(0, 1)))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=125)
    plt.close(fig)


@torch.no_grad()
def infer_tta8(model, noisy: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    """D4 = 4 obroty x 2 odbicia; odwracamy transformacje i usredniamy w FP32."""
    x = noisy.unsqueeze(0).to(device)
    restored = []
    for flip in (False, True):
        z = x.flip(-1) if flip else x
        for k in range(4):
            transformed = z.rot90(k, dims=(-2, -1))
            pred = model(transformed, sigma).float().rot90(-k, dims=(-2,-1))
            if flip:
                pred = pred.flip(-1)
            restored.append(pred)
    return torch.stack(restored, dim=0).mean(0).squeeze(0).clamp(0, 1).cpu()


def run(args):
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DnCNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint.get("eval_state_dict", checkpoint["state_dict"]))
    print("Ewaluowane wagi:", checkpoint.get("eval_weights", "raw (V2)"))
    model.eval()
    patch_size = int(checkpoint.get("train_config", {}).get("patch_size", 256))
    if patch_size != 256 and not args.allow_incomplete:
        raise ValueError(f"Checkpoint trenowano na patch_size={patch_size}. "
                         "Raport SIGK wymaga 256x256 (dla smoke-testu --allow-incomplete).")
    ds = DIV2KDenoising(args.data_dir, args.split, size=patch_size, augment="none", seed=args.seed)
    expected = 50 if args.split in ("val", "test") else 800
    if len(ds.paths) != expected and not args.allow_incomplete:
        raise RuntimeError(f"Niekompletny zbior {args.split}: {len(ds.paths)}/{expected}")
    if args.limit is not None and not args.allow_incomplete:
        raise ValueError("--limit wymaga --allow-incomplete (wynik nie jest pelnym testem)")
    total = min(len(ds), args.limit) if args.limit else len(ds)
    lpips_metric = None if args.skip_lpips else build_lpips(device)
    sigmas = list(SIGMAS)
    if args.extra_sigma is not None:
        if not 0 < args.extra_sigma <= 0.1:
            raise ValueError("--extra-sigma musi byc z (0,0.1]")
        if args.extra_sigma not in sigmas:
            sigmas.append(args.extra_sigma)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tuned = None
    if args.bilateral_config:
        config = json.loads(Path(args.bilateral_config).read_text(encoding="utf-8"))
        tuned = config["best"]
        print(f"Strojony baseline z VAL: {args.bilateral_config}")
    per_image = []
    groups = defaultdict(lambda: defaultdict(list))
    visuals = []
    for i in tqdm(range(total), desc=f"Ewaluacja {args.split}"):
        _, clean, _ = ds[i]
        name = ds.paths[i].stem
        for sigma in sigmas:
            for repeat in range(args.noise_repeats):
                # Te same inputy, identyczna realizacja szumu, dla wszystkich metod.
                noisy = fixed_noise(clean, sigma, args.seed, i, repeat)
                params = tuned.get(f"sigma_{sigma:.2f}") if tuned else None
                spatial = params["sigma_spatial"] if params else args.bilateral_spatial
                color_factor = params["sigma_color_multiplier"] if params else args.bilateral_color_factor
                filtered = bilateral(noisy, sigma, spatial, color_factor)
                with torch.no_grad():
                    output = model(noisy.unsqueeze(0).to(device), sigma).squeeze(0).clamp(0, 1).cpu()
                outputs = {"Noisy (bez filtra)": noisy, "Bilateral (skimage)": filtered,
                           "DnCNN V3": output}
                if getattr(args, 'tta', False):
                    outputs["DnCNN V3 + TTA8"] = infer_tta8(model, noisy, sigma, device)
                for method, pred in outputs.items():
                    values = metrics(pred, clean, lpips_metric, device)
                    for key, value in values.items():
                        if value is not None:
                            groups[(sigma, method)][key].append(value)
                    per_image.append({"image_id": name, "sigma": sigma, "noise_repeat": repeat,
                                      "method": method, **values})
                if repeat == 0 and i < args.examples:
                    visuals.append((f"{name} / σ={sigma}", [noisy, filtered, output, clean]))

    # Agregacja: srednia z PSNR/SSIM/LPIPS per (obraz, realizacja szumu).
    summary = []
    for sigma in sigmas:
        methods = ("Noisy (bez filtra)", "Bilateral (skimage)", "DnCNN V3")
        if getattr(args, 'tta', False):
            methods += ("DnCNN V3 + TTA8",)
        for method in methods:
            values = groups[(sigma, method)]
            row = dict(sigma=sigma, method=method, images=total, noise_repeats=args.noise_repeats,
                       PSNR=float(np.mean(values["PSNR"])), SSIM=float(np.mean(values["SSIM"])),
                       LPIPS=float(np.mean(values["LPIPS"])) if values["LPIPS"] else None)
            summary.append(row)
            lpips_text = "n/a" if row["LPIPS"] is None else f"{row['LPIPS']:.4f}"
            print(f"σ={sigma:.2f} | {method:22} | PSNR {row['PSNR']:.3f} | "
                  f"SSIM {row['SSIM']:.4f} | LPIPS {lpips_text}")
    report = {
        "dataset": "DIV2K", "split": args.split, "split_protocol": "train: 0001-0800, val: 0801-0850, test: 0851-0900",
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_eval_weights": checkpoint.get("eval_weights", "raw"),
        "tta_8_enabled": bool(getattr(args, 'tta', False)),
        "train_config": checkpoint.get("train_config", {}),
        "patch_size": patch_size, "seed": args.seed, "evaluated_image_ids": [p.stem for p in ds.paths[:total]],
        "noise_repeats": args.noise_repeats, "sigma_values": sigmas,
        "bilateral_params": {"sigma_spatial": args.bilateral_spatial,
                             "sigma_color_multiplier": args.bilateral_color_factor,
                             "tuned_on_val": tuned is not None, "per_sigma": tuned},
        "lpips_enabled": not args.skip_lpips, "protocol_complete": total == expected and patch_size == 256 and not args.skip_lpips,
        "results": summary,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for filename, rows, keys in (
        ("results.csv", summary, ["sigma", "method", "images", "noise_repeats", "PSNR", "SSIM", "LPIPS"]),
        ("per_image.csv", per_image, ["image_id", "sigma", "noise_repeat", "method", "PSNR", "SSIM", "LPIPS"]),
    ):
        with (out / filename).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    save_visuals(visuals, out / "examples.png")
    # Gotowa tabela Markdown do dokumentacji eksperymentow (bez zmyslonych wynikow).
    table = [
        "# SIGK P1 — zadanie 2.2: wyniki odszumiania",
        "",
        f"- Dane: DIV2K, protokol train 0001-0800 / val 0801-0850 / test 0851-0900.",
        f"- Raportowany split: {args.split}; obrazow: {total}; patch: {patch_size}x{patch_size}.",
        f"- Trening: augment={checkpoint.get('train_config', {}).get('augment', 'nieznany')}, "
        f"epoka checkpointu={checkpoint['epoch']}.",
        f"- Realizacji szumu na obraz: {args.noise_repeats}, seed: {args.seed}.",
        f"- Baseline: skimage.denoise_bilateral, strojony na walidacji: {tuned is not None}.",
        f"- LPIPS: {'pominiety (RAPORT NIEKOMPLETNY)' if args.skip_lpips else 'AlexNet, normalize=True'}.",
        "",
        "| σ | Metoda | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---:|---|---:|---:|---:|",
    ]
    for entry in summary:
        lpips = "—" if entry["LPIPS"] is None else f"{entry['LPIPS']:.4f}"
        table.append(f"| {entry['sigma']:.2f} | {entry['method']} | "
                     f"{entry['PSNR']:.3f} | {entry['SSIM']:.4f} | {lpips} |")
    table.extend(["", "![Wizualizacje metod](examples.png)", "",
                  "Metryki to srednia wynikow liczonych osobno na kazdym obrazie "
                  "(i powtorzeniu szumu). Wersja raportu generowana automatycznie; "
                  "uzupelnij wnioski interpretacyjne po pelnym treningu."])
    (out / "report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Wyniki w: {out.resolve()}")
    if args.skip_lpips:
        print("UWAGA: LPIPS pominięto: wyniki NIE sa kompletne wzgledem wymagan SIGK.")
    return report


def parser():
    p = argparse.ArgumentParser(description="SIGK 2.2: ewaluacja DnCNN vs bilateral, caly split test")
    p.add_argument("--data-dir", type=Path, default=Path("data/DIV2K"))
    p.add_argument("--checkpoint", type=Path, default=Path("runs/dncnn_geometric/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/dncnn_geometric/evaluation_test"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--noise-repeats", type=int, default=1)
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    p.add_argument("--bilateral-spatial", type=float, default=2.0)
    p.add_argument("--bilateral-color-factor", type=float, default=2.0)
    p.add_argument("--bilateral-config", type=Path, default=None,
                   help="JSON z tune_baseline (strojenie tylko na walidacji)")
    p.add_argument("--extra-sigma", type=float, default=None,
                   help="opcjonalny OOD test dla nieuczonej wartosci sigma, np. 0.02")
    p.add_argument("--tta", action="store_true", help="D4 TTA, 8 predykcji / obraz")
    p.add_argument("--skip-lpips", action="store_true", help="TYLKO smoke-test; nie do raportu")
    p.add_argument("--limit", type=int, default=None, help="TYLKO smoke-test")
    p.add_argument("--allow-incomplete", action="store_true", help="TYLKO smoke-test")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.noise_repeats < 1:
        parser().error("--noise-repeats musi byc >= 1")
    run(args)
