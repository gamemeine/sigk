"""Referencja z polecenia SIGK: bilateral filter ze skimage."""
import numpy as np
import torch
from skimage.restoration import denoise_bilateral


def bilateral(noisy: torch.Tensor, sigma: float, sigma_spatial: float = 2.0,
              sigma_color_multiplier: float = 2.0) -> torch.Tensor:
    """Bilateral RGB, channel_axis=-1, wspolczynnik intensywnosci zalezy od sigma.

    Parametry baseline'u musza byc zamrozone przed ocena na test, by uniknac
    tuningu na zbiorze testowym. Domyslne sigma_color=2*sigma.
    """
    inp = noisy.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    filtered = denoise_bilateral(inp, sigma_color=sigma_color_multiplier * sigma,
                                 sigma_spatial=sigma_spatial, channel_axis=-1)
    return torch.from_numpy(np.asarray(filtered, dtype=np.float32).copy()).permute(2, 0, 1).clamp(0, 1)
