import numpy as np
import math
import torch

from typing import Union
from torch import Tensor
from pathlib import Path

from config import *
from metrics import joint_accuracy, joint_iou
from evaluation import EvaluationResult


def _get_patch_coords(
    img_shape: tuple[int, int], patch_size: int, offset: int, overlap: int
) -> list[tuple[int, int]]:
    """
    Arguments:
        img_shape (tuple[int, int]): Image shape.
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int): Overlap number in pixels.

    Returns:
        coords (list[tuple[int, int]]): List of patch coordinates.
    """

    H, W = img_shape
    step = patch_size - 2 * offset - overlap
    nh = math.ceil((H - 2 * offset) / step)
    nw = math.ceil((W - 2 * offset) / step)

    coords = []
    for i in range(nh):
        y = min(i * step, H - patch_size)
        for j in range(nw):
            x = min(j * step, W - patch_size)
            coords.append((y, x))
    return coords


def split_into_patches(
    image: Tensor, patch_size: int, offset: int, overlap: Union[int, float]
) -> list[Tensor]:
    """
    Splits image into patches.

    Arguments:
        image (Tensor): Source image of size (..., H, W).
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels

    Returns:
        patchs (list[Tensor]): List of extracted patches.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    coords = _get_patch_coords(image.shape[-2:], patch_size, offset, overlap)
    patches = []
    for coord in coords:
        y, x = coord
        patch = image[..., y : y + patch_size, x : x + patch_size]
        patches.append(patch)
    return patches


def combine_from_patches(
    patches: list[Tensor],
    patch_size: int,
    offset: int,
    overlap: Union[int, float],
    src_shape: tuple[int, int],
    device: torch.device,
) -> Tensor:
    """
    Combines patches back into the image.

    Arguments:
        patches (list[Tensor]): List of patches of size (n_classes, patch_size, patch_size)
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels
        src_shape (list[int, int]): Source image shape.

    Returns:
        image (Tensor): Combined image.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    image = torch.zeros(patches[0].shape[:-2] + src_shape, device=device, dtype=torch.float32)
    density = torch.zeros(*src_shape, device=device, dtype=torch.float32)
    coords = _get_patch_coords(src_shape, patch_size, offset, overlap)
    for i, coord in enumerate(coords):
        y, x = coord
        y0, y1 = y + offset, y + patch_size - offset
        x0, x1 = x + offset, x + patch_size - offset
        image[..., y0:y1, x0:x1] += patches[i][
            ...,
            offset : patch_size - offset,
            offset : patch_size - offset,
        ]
        density[..., y0:y1, x0:x1] += 1
    density[density == 0] = 1
    image /= density
    return image


def one_hot(mask: Tensor, n_classes: int) -> Tensor:
    """
    Applies one hot encoding

    Arguments:
        mask (Tensor): Input tensor with size (H, W) with elements in range [0, n_classes)
        n_classes (int): Total number of classes

    Return:
        one_hot_mask (Tensor): Output tensor with size (n_classes, H, W)
    """

    new_mask = []
    for i in range(n_classes):
        new_mask.append(mask == i)
    
    one_hot_mask = torch.stack(new_mask, dim=0).to(dtype=torch.float32)
    return one_hot_mask


def prepare_experiment(output_path: Path) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    dirs = list(output_path.iterdir())
    dirs = [d for d in dirs if d.name.startswith("exp_")]
    exp_id = max(int(d.name.split("_")[1]) for d in dirs) + 1 if dirs else 1
    exp_path = output_path / f"exp_{exp_id}"
    exp_path.mkdir()
    return exp_path


def preprocess_mask(mask: Tensor) -> Tensor:
    """
    Squeeze codes in mask and apply one hot encoding

    Arguments:
        mask (Tensor): Input mask of size (H, W)
    Returns:
        target (Tensor): One hot encoded mask of shape (n_classes, H, W)
    """

    """ Squeeze mask """
    squeezed_mask = torch.zeros_like(mask)
    for code, squeezed_code in codes2squeezed_codes.items():
        squeezed_mask[mask == code] = squeezed_code
    
    return one_hot(squeezed_mask, len(present_class_codes))


def write_metrics(file: object, eval_res: EvaluationResult, description: str) -> None:
    file.write(eval_res.to_str(description))
