from datetime import datetime
import os
from pathlib import Path
from random import Random
from typing import Callable

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from p1.inpainting.baseline import inpaint
from p1.inpainting.data import DIV2KDataset


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


def print_results(results: dict[str, dict[str, float]], output_path: Path, caption=None) -> None:
    table = Table("Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓")
    for method, scores in results.items():
        table.add_row(
            method,
            f"{scores['PSNR']:.4f}",
            f"{scores['SSIM']:.6f}",
            f"{scores['LPIPS']:.6f}",
        )

    table.caption = caption
    Console().print(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        import json
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
    parser.add_argument("--examples", type=int, default=15)


    args = parser.parse_args()

    for cut_size in [3, 32]:
        results = {}
        test_dataset = DIV2KDataset(root=args.data_dir, split="valid", seed=args.seed, cut_size=cut_size)

        # no inpainting
        results[f"no_inpaint_{cut_size}_x_{cut_size}"] = evaluate(test_dataset, lambda img, mask: img)

        # opencv Telea
        results[f"opencv_inpaint_telea_{cut_size}_x_{cut_size}"] = evaluate(test_dataset, inpaint)

        # print and plot results
        results_path = args.output_dir / f"results_{cut_size}x{cut_size}.json"
        examples_path = args.output_dir / f"examples_{cut_size}x{cut_size}.png"

        print_results(
            {name: value[0] for name, value in results.items()},
            results_path,
            caption=f"Results for cut size {cut_size}x{cut_size}",
        )
        plot_predictions(
            {name: value[1] for name, value in results.items()},
            examples_path,
            seed=args.seed,
            samples=args.examples,
        )
