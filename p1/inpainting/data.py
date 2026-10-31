from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import ceil
from pathlib import Path
from random import Random
from shutil import copyfileobj
from zipfile import ZipFile

import cv2
import requests
import torch
from torch.utils.data import Dataset

from p1.inpainting.utils import random_cut


def _preprocess_image(image_path, seed, output_dir, target_res, crop_strategy):
    image = cv2.imread(str(image_path))
    assert image is not None, f"Cannot read image: {image_path}"

    height, width = image.shape[:2]
    target_width, target_height = target_res
    scale = max(target_width / width, target_height / height)
    width, height = ceil(width * scale), ceil(height * scale)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if crop_strategy == "random":
        rng = Random(seed)
        top = rng.randint(0, height - target_height)
        left = rng.randint(0, width - target_width)
    else:
        top = (height - target_height) // 2
        left = (width - target_width) // 2

    image = image[top : top + target_height, left : left + target_width]
    output_path = output_dir / image_path.name
    assert cv2.imwrite(str(output_path), image), "Cannot write image"


def download_and_preprocess(
    output_dir: Path,
    target_res=(256, 256),
    crop_strategy="random",
    seed: int = 42,
    num_workers: int = 4,
):
    output_dir = Path(output_dir)
    raw_root = output_dir / "raw" / "DIV2K"
    raw_root.mkdir(parents=True, exist_ok=True)
    rng = Random(seed)

    for split in ("train", "valid"):
        archive_path = raw_root / f"DIV2K_{split}_HR.zip"
        if not archive_path.exists():
            url = f"https://data.vision.ee.ethz.ch/cvl/DIV2K/{archive_path.name}"
            print(f"Downloading {url}", flush=True)
            download_path = archive_path.with_suffix(".zip.part")
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with download_path.open("wb") as destination:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        destination.write(chunk)
            download_path.replace(archive_path)

        raw_dir = raw_root / split
        processed_dir = output_dir / "processed" / "DIV2K" / split
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".png"):
                    continue
                image_path = raw_dir / Path(member.filename).name
                if not image_path.exists() or image_path.stat().st_size != member.file_size:
                    with archive.open(member) as source, image_path.open("wb") as destination:
                        copyfileobj(source, destination)
                image_paths.append(image_path)

        image_paths.sort()
        preprocess = partial(
            _preprocess_image,
            output_dir=processed_dir,
            target_res=target_res,
            crop_strategy=crop_strategy,
        )
        seeds = [rng.getrandbits(32) for _ in image_paths]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in executor.map(preprocess, image_paths, seeds):
                pass
        print(f"Processed {len(image_paths)} {split} images: {processed_dir}", flush=True)


class DIV2KDataset(Dataset):
    def __init__(
        self, root: Path, split: str, num_samples: int | None = None, seed: int = 42,
        cut_size: int | list[int] = 3, random_images: bool = True,
    ):
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive")
        image_dir = Path(root) / "processed" / "DIV2K" / split
        self.image_paths = sorted(image_dir.glob("*.png"))

        self.num_samples = num_samples
        self.random_images = random_images
        self.cut_sizes = [cut_size] if isinstance(cut_size, int) else cut_size
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = Random(seed)

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        image_path = (
            self.rng.choice(self.image_paths) if self.random_images
            else self.image_paths[index % len(self.image_paths)]
        )
        image = cv2.imread(str(image_path))
        assert image is not None, f"Cannot read image: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cut_size = self.rng.choice(self.cut_sizes)
        cut_seed = self.rng.getrandbits(32)
        cut_image, mask = random_cut(image, cut_size, cut_seed)

        cut_tensor = torch.from_numpy(cut_image).permute(2, 0, 1).float() / 255
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255
        target_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255

        return cut_tensor, mask_tensor, target_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="download")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed data",
    )
    parser.add_argument(
        "--target-res",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Target resolution: width height",
    )
    parser.add_argument(
        "--crop-strategy",
        type=str,
        default="random",
        choices=["random", "center"],
        help="Cropping strategy to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    args = parser.parse_args()

    download_and_preprocess(
        Path(args.output_dir),
        target_res=tuple(args.target_res),
        crop_strategy=args.crop_strategy,
        seed=args.seed,
        num_workers=args.num_workers,
    )
