from random import Random

import numpy as np


def random_cut(img: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    if size <= 0 or size > min(height, width):
        raise ValueError("size must be positive and fit within the image")

    rng = Random(seed)
    top = rng.randint(0, height - size)
    left = rng.randint(0, width - size)

    cut_img = img.copy()
    cut_img[top : top + size, left : left + size] = 0

    # OpenCV uses nonzero mask values for pixels to inpaint.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + size, left : left + size] = 255

    return cut_img, mask
