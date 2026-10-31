import json
import os
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from scipy.stats import sem
from torch.utils.data import Dataset
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset
from p1.inpainting.models.baseline import inpaint


def evaluate(test_ds: Dataset, inpaint_fun: Callable):
    metrics = {
        "PSNR": PeakSignalNoiseRatio(data_range=1.0),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0),
        "LPIPS": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True),
    }

    predictions = []
    for cut_image, mask, target in test_ds:
        target = target.unsqueeze(0)
        prediction = inpaint_fun(cut_image, mask).unsqueeze(0)

        for metric in metrics.values():
            metric.update(prediction, target)

        predictions.append((cut_image, mask, target, prediction))

    return {name: metric.compute().item() for name, metric in metrics.items()}, predictions


def summarize_results(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in runs[0]:
        values = [run[metric] for run in runs]
        summary[metric] = {"mean": mean(values)}
        if len(values) > 1:
            summary[metric]["se"] = float(sem(values))
    return summary


def print_results(results: dict[str, dict[str, dict[str, float]]], output_path: Path, caption=None) -> None:
    table = Table("Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓")
    for method, scores in results.items():
        cells = []
        for metric in ("PSNR", "SSIM", "LPIPS"):
            score = scores[metric]
            cell = f"{score['mean']:.4f}"
            if "se" in score:
                cell += f" ± {score['se']:.4f}"
            cells.append(cell)
        table.add_row(method, *cells)

    table.caption = caption
    Console().print(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def plot_predictions(predictions: dict, output_path: Path, samples: int = 10, seed: int = 42) -> Path:
    examples = next(iter(predictions.values()))
    indices = Random(seed).sample(range(len(examples)), min(samples, len(examples)))
    titles = ["sample", *predictions, "original"]
    figure, axes = plt.subplots(
        len(indices),
        len(titles),
        figsize=(3 * len(titles), 3 * len(indices)),
        squeeze=False,
        layout="constrained",
    )

    for row, index in enumerate(indices):
        cut_image, _, target, _ = examples[index]
        images = [cut_image, *(values[index][3] for values in predictions.values()), target]
        for column, image in enumerate(images):
            axes[row, column].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])

    os.makedirs(output_path.parent, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("p1/output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--examples", type=int, default=15)

    args = parser.parse_args()

    seeds = Random(args.seed).sample(range(1_000_000), args.repeats)
    methods = {"no_inpaint": lambda img, mask: img, "opencv_inpaint_telea": inpaint}
    for cut_size in [3, 32]:
        results = {name: [] for name in methods}
        predictions = {}
        for repetition, seed in tqdm(enumerate(seeds), total=args.repeats, desc=f"Cut {cut_size}x{cut_size}", leave=False):
            test_dataset = DIV2KDataset(root=args.data_dir, split="valid", seed=seed, cut_size=cut_size)
            for name, method in methods.items():
                scores, examples = evaluate(test_dataset, method)
                results[name].append(scores)
                if repetition == 0:
                    predictions[name] = examples

        # print and plot results
        results_path = args.output_dir / f"results_{cut_size}x{cut_size}.json"
        examples_path = args.output_dir / f"examples_{cut_size}x{cut_size}.png"

        print_results(
            {name: summarize_results(runs) for name, runs in results.items()},
            results_path,
            caption=(
                f"Cut {cut_size}x{cut_size}, single seed" if args.repeats == 1
                else f"Cut {cut_size}x{cut_size}, mean ± SE over {args.repeats} seeds"
            ),
        )
        plot_predictions(
            predictions,
            examples_path,
            seed=args.seed,
            samples=args.examples,
        )
