import cv2
import numpy as np
import torch
from torch import nn


class NoInpaintModel(nn.Module):
    def predict(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return image


class TeleaModel(nn.Module):
    def __init__(self, radius: float = 3.0):
        super().__init__()
        self.radius = radius

    def predict(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).round().astype(np.uint8)
        mask_array = mask.squeeze(0).numpy().astype(np.uint8)

        result = cv2.inpaint(image, mask_array, self.radius, cv2.INPAINT_TELEA)

        return torch.from_numpy(result).permute(2, 0, 1).float() / 255
