import json
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import sem
from torch.utils.data import Dataset
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset


@torch.no_grad()
def evaluate(test_ds: Dataset, device: str = "cpu", model=None):
    metrics = {
        "PSNR": PeakSignalNoiseRatio(data_range=1.0),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0),
        "LPIPS": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True),
    }
    hole_metrics = {
        "PSNR_hole": PeakSignalNoiseRatio(data_range=1.0),
        "MAE_hole": MeanAbsoluteError(),
    }
    for metric in (*metrics.values(), *hole_metrics.values()):
        metric.to(device).eval()
    if model is not None:
        model.to(device).eval()

    predictions = []
    for cut_image, mask, target in test_ds:
        if model is None:
            image = (cut_image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            hole = mask.squeeze(0).numpy().astype(np.uint8)
            restored = cv2.inpaint(image, hole, 3.0, cv2.INPAINT_TELEA)
            prediction = torch.from_numpy(restored).permute(2, 0, 1).float() / 255
        else:
            inputs = torch.cat([cut_image, mask]).unsqueeze(0).to(device)
            prediction = model(inputs).squeeze(0).cpu()
            prediction = cut_image * (1 - mask) + prediction * mask
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        metric_prediction = prediction.to(device)
        metric_target = target.to(device)
        hole_mask = mask.to(device).bool().expand_as(metric_target)

        for metric in metrics.values():
            metric.update(metric_prediction, metric_target)
        for metric in hole_metrics.values():
            metric.update(metric_prediction[hole_mask], metric_target[hole_mask])

        predictions.append((cut_image, prediction.squeeze(0), target.squeeze(0)))

    scores = {name: metric.compute().item() for name, metric in (metrics | hole_metrics).items()}
    return scores, predictions


def summarize_results(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in runs[0]:
        values = [run[metric] for run in runs]
        summary[metric] = {"mean": mean(values)}
        if len(values) > 1:
            summary[metric]["se"] = float(sem(values))
    return summary


def print_results(results: dict[str, dict[str, dict[str, float]]], output_path: Path, caption=None) -> None:
    groups = (
        ("Image", {"PSNR": "PSNR ↑", "SSIM": "SSIM ↑", "LPIPS": "LPIPS ↓"}),
        ("Hole", {"PSNR_hole": "PSNR ↑", "MAE_hole": "MAE ↓"}),
    )
    table = Table(caption=caption)
    table.add_column("Method", vertical="middle")
    for title, _ in groups:
        table.add_column(title, justify="center")

    for method, scores in results.items():
        sections = []
        for _, columns in groups:
            section = Table(*columns.values(), box=None, padding=(0, 1), pad_edge=False)
            cells = []
            for metric in columns:
                score = scores[metric]
                cell = f"{score['mean']:.4f}"
                if "se" in score:
                    cell += f" ± {score['se']:.4f}"
                cells.append(cell)
            section.add_row(*cells)
            sections.append(section)
        table.add_row(method, *sections)
    Console().print(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def plot_predictions(predictions: dict, output_path: Path, samples: int = 10, seed: int = 42) -> Path:
    examples = next(iter(predictions.values()))
    indices = Random(seed).sample(range(len(examples)), min(samples, len(examples)))
    titles = ["input", *predictions, "original"]
    figure, axes = plt.subplots(
        len(indices),
        len(titles),
        figsize=(3 * len(titles), 3 * len(indices)),
        squeeze=False,
        layout="constrained",
    )

    for row, index in enumerate(indices):
        images = [examples[index][0], *(values[index][1] for values in predictions.values()), examples[index][2]]
        for column, image in enumerate(images):
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run_evaluation(data_dir, output_dir, seed=42, repeats=1, examples=15, samples=None, device="cpu", model=None):
    output_dir = Path(output_dir)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    methods = {"opencv_inpaint_telea": None}
    if model is not None:
        methods["residual_unet"] = model
    evaluation_config = {
        "data_dir": data_dir, "output_dir": output_dir,
        "seed": seed, "repeats": repeats, "examples": examples, "samples": samples, "device": device,
        "methods": list(methods),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(json.dumps(evaluation_config, indent=2, default=str))

    seeds = Random(seed).sample(range(1_000_000), repeats)
    results_by_size = {}
    for cut_size in [3, 32]:
        runs = {name: [] for name in methods}
        predictions = {}
        for repetition, cut_seed in tqdm(
            enumerate(seeds), total=repeats, desc=f"Cut {cut_size}x{cut_size}", leave=False,
        ):
            test_dataset = DIV2KDataset(
                root=data_dir, split="valid", num_samples=samples, seed=cut_seed,
                cut_size=cut_size, random_images=False,
            )
            for name, method in methods.items():
                scores, examples_for_seed = evaluate(test_dataset, device=device, model=method)
                runs[name].append(scores)
                if repetition == 0:
                    predictions[name] = examples_for_seed

        # print and plot results
        results_path = output_dir / f"results_{cut_size}x{cut_size}.json"
        examples_path = output_dir / f"examples_{cut_size}x{cut_size}.png"

        summary = {name: summarize_results(method_runs) for name, method_runs in runs.items()}
        results_by_size[cut_size] = summary
        print_results(
            summary,
            results_path,
            caption=(
                f"Cut {cut_size}x{cut_size}, single seed" if repeats == 1
                else f"Cut {cut_size}x{cut_size}, mean ± SE over {repeats} seeds"
            ),
        )
        plot_predictions(
            predictions,
            examples_path,
            seed=seed,
            samples=examples,
        )

    return results_by_size


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
        default=Path("p1/output/results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--examples", type=int, default=15)
    parser.add_argument("--samples", type=int, help="Number of evaluation samples; default: all images")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    run_evaluation(**vars(args))
