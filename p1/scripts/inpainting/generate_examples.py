import argparse
from pathlib import Path
from random import Random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from p1.inpainting.data import DIV2KDataset

plt.switch_backend("Agg")


def method_name(config):
    model = config.model._target_.split(".")[-1]
    name = {"ResidualUNet": "unet", "PartialUNet": "pconv"}.get(model, model)
    loss = config.get("loss", {})
    if str(loss.get("_target_", "")).endswith("PartConvLoss"):
        return f"{name}_paper"
    if loss.get("style_weight", 0):
        return f"{name}_style"
    gradient = loss.get("gradient_weight", 0)
    perceptual = loss.get("perceptual_weight", 0)
    if not gradient and not perceptual:
        return f"{name}_l1"
    if name == "pconv" or gradient != 0.1:
        return f"{name}_g{gradient:g}".replace(".", "_")
    return f"{name}_p{perceptual:g}".replace(".", "_")


def plot_examples(columns, masks, cut_size, output_path, detail=False):
    figure, axes = plt.subplots(
        len(masks), len(columns), figsize=(2.4 * len(columns), 2.4 * len(masks)),
        squeeze=False, layout="constrained",
    )
    for row, mask in enumerate(masks):
        positions = mask[0].nonzero()
        top, left = positions.min(dim=0).values.tolist()
        height, width = mask.shape[-2:]
        crop_size = min(max(2 * cut_size, 24), height, width)
        crop_top = max(0, min(top + cut_size // 2 - crop_size // 2, height - crop_size))
        crop_left = max(0, min(left + cut_size // 2 - crop_size // 2, width - crop_size))
        for column, (name, images) in enumerate(columns.items()):
            axis = axes[row, column]
            image = images[row]
            if detail:
                image = image[:, crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
            axis.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(name, fontsize=10)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


@torch.inference_mode()
def generate_examples(experiments, data_dir, output_dir, examples=5, seed=42, device="auto", names=None):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [OmegaConf.load(path / "config.yaml") for path in experiments]
    names = names or [method_name(config) for config in configs]
    if len(names) != len(experiments) or len(set(names)) != len(names):
        raise ValueError("Provide one unique --names label per experiment")

    # Match the first evaluation run and example selection in evaluate.py.
    mask_seed = Random(seed).sample(range(1_000_000), 1)[0]
    datasets = {
        size: DIV2KDataset(data_dir, "valid", seed=mask_seed, crop_size=256, cut_size=size, random_images=False)
        for size in (3, 32)
    }
    indices = Random(seed).sample(range(len(datasets[3])), min(examples, len(datasets[3])))
    batches = {}
    columns = {}
    for size, dataset in datasets.items():
        cut, masks, targets = (torch.stack(values) for values in zip(*(dataset[index] for index in indices)))
        restored = []
        for image, mask in zip(cut, masks):
            image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            result = cv2.inpaint(image, mask[0].numpy().astype(np.uint8), 3.0, cv2.INPAINT_TELEA)
            restored.append(torch.from_numpy(result).permute(2, 0, 1).float() / 255)
        batches[size] = cut, masks, targets
        columns[size] = {"input": cut, "telea": torch.stack(restored)}

    for path, config, name in tqdm(list(zip(experiments, configs, names)), desc="Models"):
        checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        model = instantiate(config.model)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        for size, (cut, masks, _) in batches.items():
            prediction = model(torch.cat([cut, masks], dim=1).to(device)).cpu()
            columns[size][name] = cut * (1 - masks) + prediction * masks
        del model, checkpoint

    paths = []
    for size, (_, masks, targets) in batches.items():
        columns[size]["gt"] = targets
        for detail in (False, True):
            suffix = "_detail" if detail else ""
            path = output_dir / f"examples_{size}x{size}{suffix}.png"
            plot_examples(columns[size], masks, size, path, detail=detail)
            paths.append(path)
            print(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("p1/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("p1/figures/inpainting"))
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--names", nargs="+", help="Optional unique column labels in experiment order")
    args = parser.parse_args()
    generate_examples(**vars(args))
