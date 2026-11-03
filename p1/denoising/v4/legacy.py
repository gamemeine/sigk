"""DnCNN przewidujacy szum (residual learning), opcjonalnie conditionowany sigma."""
import torch
from torch import nn


class DnCNN(nn.Module):
    """RGB + mapa sigma/0.03 -> przewidywany SZUM ZNORMALIZOWANY RGB; wynik = obraz - sigma*szum.

    Poza ostatnia warstwa kazdy blok ma splot 3x3, a bloki wewnetrzne
    dodatkowo BatchNorm + ReLU. DnCNN z 17 warstwami i 64 kanalami
    jest sensowna konfiguracja wyjsciowa (to nie jest dokladna reprodukcja wag).
    """

    def __init__(self, depth: int = 17, features: int = 64, conditioned: bool = True):
        super().__init__()
        if depth < 3 or features < 4:
            raise ValueError("depth >= 3, features >= 4")
        self.depth, self.features, self.conditioned = depth, features, conditioned
        in_ch = 4 if conditioned else 3
        layers: list[nn.Module] = [nn.Conv2d(in_ch, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)
        # Z poczatku siec jest bliska identycznosci (szum=0).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, noisy: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        if self.conditioned:
            if not torch.is_tensor(sigma):
                sigma = torch.tensor(sigma, device=noisy.device, dtype=noisy.dtype)
            sigma = sigma.to(device=noisy.device, dtype=noisy.dtype)
            if sigma.ndim == 0:
                sigma = sigma.expand(noisy.shape[0])
            if sigma.ndim != 1 or sigma.shape[0] != noisy.shape[0]:
                raise ValueError("sigma musi byc skalarem lub tensorem [batch]")
            sigma_map = (sigma / 0.03).view(-1, 1, 1, 1).expand(-1, -1, noisy.shape[2], noisy.shape[3])
            x = torch.cat((noisy, sigma_map), dim=1)
        else:
            x = noisy
        return noisy - self.net(x) * sigma.view(-1, 1, 1, 1)
