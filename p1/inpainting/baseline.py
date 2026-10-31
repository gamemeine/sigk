import cv2
import numpy as np
import torch


def inpaint(img: torch.Tensor, mask: torch.Tensor, radius: float = 3.0) -> torch.Tensor:
    image = img.permute(1, 2, 0).numpy()
    image = (image * 255).round().astype(np.uint8)
    mask_array = mask.squeeze(0).numpy().astype(np.uint8)

    result = cv2.inpaint(image, mask_array, radius, cv2.INPAINT_TELEA)

    return torch.from_numpy(result).permute(2, 0, 1).float() / 255
