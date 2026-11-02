from math import ceil
from pathlib import Path
from random import Random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from p1.inpainting.utils import random_cut


class DIV2KDataset(Dataset):
    def __init__(
        self, root: Path, split: str, num_samples: int | None = None, seed: int = 42,
        cut_size: int | list[int] = 3, random_images: bool = True, crop_size: int = 256,
        augment: bool = False,
    ):
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive")
        image_dir = Path(root) / "raw" / "DIV2K" / split
        self.image_paths = sorted(image_dir.glob("*.png"))

        self.num_samples = num_samples
        self.random_images = random_images
        self.crop_size = crop_size
        self.augment = augment
        self.cut_sizes = [cut_size] if isinstance(cut_size, int) else cut_size
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Each index fixes the image, crop and hole independently of workers and access order.
        rng = Random(f"{self.seed}:{index}")
        image_path = (
            rng.choice(self.image_paths) if self.random_images
            else self.image_paths[index % len(self.image_paths)]
        )
        image = cv2.imread(str(image_path))
        assert image is not None, f"Cannot read image: {image_path}"

        height, width = image.shape[:2]
        scale = self.crop_size / min(height, width)
        height, width = ceil(height * scale), ceil(width * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        top = rng.randint(0, height - self.crop_size)
        left = rng.randint(0, width - self.crop_size)
        image = image[top:top + self.crop_size, left:left + self.crop_size]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            if rng.random() < 0.5:
                image = np.flip(image, axis=1)
            if rng.random() < 0.5:
                image = np.flip(image, axis=0)
            image = np.ascontiguousarray(np.rot90(image, rng.randrange(4)))

        cut_size = rng.choice(self.cut_sizes)
        cut_seed = rng.getrandbits(32)
        cut_image, mask = random_cut(image, cut_size, cut_seed)

        cut_tensor = torch.from_numpy(cut_image).permute(2, 0, 1).float() / 255
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255
        target_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255

        return cut_tensor, mask_tensor, target_tensor
