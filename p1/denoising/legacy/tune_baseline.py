"""Uczciwe strojenie baseline'u bilateralnego TYLKO na walidacji (nie test)."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from .baseline import bilateral
from .data import DIV2KDenoising, SIGMAS, fixed_noise
from .evaluate import tensor_to_hwc


def run(args):
    dataset = DIV2KDenoising(args.data_dir, "val", size=256, augment="none")
    if len(dataset) != 50 and not args.allow_incomplete:
        raise RuntimeError(f"Wymagane 50 obrazow do strojenia baseline'u. Jest: {len(dataset)}")
    if args.limit and not args.allow_incomplete:
        raise ValueError("--limit wymaga --allow-incomplete")
    n = min(len(dataset), args.limit) if args.limit else len(dataset)
    # Na kazdy kandydat stosujemy te same czyste obrazy i realizacje szumu.
    # Nie zapisujemy wszystkich tensorow 256x256 w RAM.
    settings = list(product(args.spatial, args.color_factors))
    results = {sigma: {setting: [] for setting in settings} for sigma in SIGMAS}
    for i in tqdm(range(n), desc="Tuning bilateral na VAL"):
        _, clean, _ = dataset[i]
        target = tensor_to_hwc(clean)
        for sigma in SIGMAS:
            noisy = fixed_noise(clean, sigma, args.seed, i)
            for setting in settings:
                pred = bilateral(noisy, sigma, sigma_spatial=setting[0], sigma_color_multiplier=setting[1])
                value = float(peak_signal_noise_ratio(target, tensor_to_hwc(pred), data_range=1.0))
                results[sigma][setting].append(value)
    best = {}
    trials = []
    for sigma in SIGMAS:
        for (spatial, color_factor), values in results[sigma].items():
            trials.append({"sigma": sigma, "sigma_spatial": spatial,
                           "sigma_color_multiplier": color_factor, "val_PSNR": float(np.mean(values))})
        winner = max((entry for entry in trials if entry["sigma"] == sigma), key=lambda row: row["val_PSNR"])
        best[f"sigma_{sigma:.2f}"] = {"sigma_spatial": winner["sigma_spatial"],
                                     "sigma_color_multiplier": winner["sigma_color_multiplier"]}
        print(f"σ={sigma:.2f}: PSNR {winner['val_PSNR']:.3f} dB, "
              f"spatial={winner['sigma_spatial']}, color_factor={winner['sigma_color_multiplier']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": "validation only: 0801-0850; PSNR optimization",
                                  "n_images": n, "seed": args.seed, "best": best,
                                  "trials": trials}, indent=2), encoding="utf-8")
    print(f"Parametry zapisane: {output}")


def parser():
    p = argparse.ArgumentParser(description="Strojenie bilateralnego baseline'u na walidacji DIV2K")
    p.add_argument("--data-dir", type=Path, default=Path("data/DIV2K"))
    p.add_argument("--output", type=Path, default=Path("runs/bilateral_val_tuned.json"))
    p.add_argument("--spatial", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--color-factors", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-incomplete", action="store_true", help="wylacznie test techniczny")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
