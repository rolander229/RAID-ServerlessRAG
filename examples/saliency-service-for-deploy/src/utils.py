import numpy as np
from typing import Union
import torch


def get_random_crop_params(image_height: int, image_width: int, min_crop_size: int) -> tuple[int, int, int]:
    crop_size = np.random.randint(min_crop_size, min(image_height, image_width))
    x = np.random.randint(image_width - crop_size + 1)
    y = np.random.randint(image_height - crop_size + 1)
    return x, y, crop_size


def get_cropped_image(im_tensor: np.array, x: int, y: int, crop_size: int) -> np.array:
    return im_tensor[y : y + crop_size, x : x + crop_size, ...]


def update_saliency_map(saliency_map: np.array, similarity: float, x: int, y: int, crop_size: int) -> None:
    saliency_map[
        y : y + crop_size,
        x : x + crop_size,
    ] += similarity


def cosine_similarity(one: Union[np.ndarray, torch.Tensor], other: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return one @ other.T / (np.linalg.norm(one) * np.linalg.norm(other))
