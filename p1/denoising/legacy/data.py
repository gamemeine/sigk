"""Odczyt DIV2K, szczelne splity, augmentacje i reprodukowalny szum Gaussa."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

SIGMAS = (0.01, 0.03)
VALID_IDS = tuple(range(801, 901))
VAL_IDS = tuple(range(801, 851))  # 0801-0850: tuning modelu
TEST_IDS = tuple(range(851, 901))  # 0851-0900: finalny, niedotykany test


def find_split_directory(root: str | Path, split: str) -> Path:
    """Akceptuje katalog root, data lub bezposredni katalog train/valid."""
    base = Path(root).expanduser()
    actual = "train" if split == "train" else "valid"
    names = (f"DIV2K_{actual}_HR", actual)
    candidates = [base]
    for prefix in (base, base / "DIV2K", base / "raw" / "DIV2K", base / "data" / "DIV2K",
                   base / "p1" / "data" / "raw" / "DIV2K"):
        candidates += [prefix / n for n in names]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(
        f"Nie znaleziono obrazow {split} w {base}. Oczekiwano DIV2K_{actual}_HR/*.png. "
        "Pobierz DIV2K: bash p1/scripts/download_data.sh lub sprawdz --data-dir."
    )


def list_images(root: str | Path, split: str) -> list[Path]:
    folder = find_split_directory(root, split)
    paths = sorted(folder.glob("*.png"))
    allowed = None
    if split == "val":
        allowed = set(VAL_IDS)
    elif split == "test":
        allowed = set(TEST_IDS)
    if allowed is not None:
        paths = [p for p in paths if p.stem.isdigit() and int(p.stem) in allowed]
    if not paths:
        raise ValueError(f"Brak obrazow PNG dla split={split} w {folder}")
    return paths


def clean_patch(image: Image.Image, size: int, rng: random.Random,
                training: bool, augment: str) -> torch.Tensor:
    """Najpierw augmentujemy obraz czysty; dopiero pozniej dodajemy szum."""
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < size:
        scale = size / min(w, h)
        image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
        w, h = image.size
    if training:
        left = rng.randrange(w - size + 1)
        top = rng.randrange(h - size + 1)
    else:
        left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    if training and augment in ("geometric", "full"):
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if rng.random() < .5:
            image = ImageOps.flip(image)
        image = image.rotate(90 * rng.randrange(4))
    if training and augment == "full":
        # Lekkie jittery, by nie uczyc sie bardzo specyficznej kolorystyki DIV2K.
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.90, 1.10))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(.90, 1.10))
        image = ImageEnhance.Color(image).enhance(rng.uniform(.90, 1.10))
    rgb = np.array(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def corrupt(clean: torch.Tensor, sigma: float, seed: int | None = None) -> torch.Tensor:
    """N(0, sigma^2) na RGB w [0,1]; obciecie do [0,1] zgodne z prezentacja obrazu."""
    if seed is None:
        noise = torch.randn_like(clean)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    return (clean + noise * sigma).clamp_(0, 1)


class DIV2KDenoising(Dataset):
    def __init__(self, root: str | Path, split: str, size: int = 256,
                 augment: str = "geometric", samples_per_epoch: int = 6400, seed: int = 42):
        if split not in ("train", "val", "test"):
            raise ValueError("split musi byc train/val/test")
        if augment not in ("none", "geometric", "full"):
            raise ValueError("augment: none/geometric/full")
        if size < 16:
            raise ValueError("size >= 16")
        self.paths = list_images(root, split)
        self.split, self.size, self.augment = split, size, augment
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch if self.split == "train" else len(self.paths)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Losowosc niezalezna od ilosci workerow i kolejnosci pobierania.
        # Indeksy/epoki sa deterministyczne, powtarzanie obrazow daje nowe cropy.
        rng = random.Random(f"{self.seed}|{self.epoch if self.split == 'train' else 0}|{index}")
        path = rng.choice(self.paths) if self.split == "train" else self.paths[index]
        with Image.open(path) as image:
            clean = clean_patch(image, self.size, rng, self.split == "train", self.augment)
        sigma = SIGMAS[rng.randrange(len(SIGMAS))] if self.split == "train" else SIGMAS[0]
        noise_seed = rng.getrandbits(63)
        noisy = corrupt(clean, sigma, noise_seed)
        return noisy, clean, torch.tensor(sigma, dtype=torch.float32)


def fixed_noise(clean: torch.Tensor, sigma: float, seed: int, image_index: int, repeat: int = 0):
    return corrupt(clean, sigma, seed + 100_003 * image_index + 10_000_019 * repeat + int(sigma * 1000))
