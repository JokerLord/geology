import numpy as np
import math
import torch

from typing import Union
from torch import Tensor


def _get_patch_coords(img_shape, patch_size, conv_offset, overlap):
    """
    Arguments:
        img_shape (tuple[int, int, int, int]): Image shape.
        patch_size (int): Patch size in pixels.
        conv_offset (int): Convolutional offset in pixels.
        overlap (int): Overlap number in pixels.

    Returns:
        coords (list[(int, int)]): List of patch coordinates.
    """

    H, W = img_shape[-2:]
    step = patch_size - 2 * conv_offset - overlap
    nh = math.ceil((H - 2 * conv_offset) / step)
    nw = math.ceil((W - 2 * conv_offset) / step)

    coords = []
    for i in range(nh):
        y = min(i * step, H - patch_size)
        for j in range(nw):
            x = min(j * step, W - patch_size)
            coords.append((y, x))
    return coords


def split_into_patches(
    image: Tensor, patch_size: int, conv_offset: int, overlap: Union[int, float]
) -> list[Tensor]:
    """
    Splits image into patches.

    Arguments:
        image (Tensor): Source image.
        patch_size (int): Patch size in pixels.
        conv_offset (int): Convolutional offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels

    Returns:
        patchs (list[np.ndarray]): List of extracted patches.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    coords = _get_patch_coords(image.shape, patch_size, conv_offset, overlap)
    patches = []
    for coord in coords:
        y, x = coord
        patch = image[..., y : y + patch_size, x : x + patch_size]
        patches.append(patch)
    return patches


def combine_from_patches(
    patches: list[Tensor],
    patch_size: int,
    conv_offset: int,
    overlap: Union[int, float],
    src_shape: tuple[int, int],
) -> Tensor:
    """
    Combines patches back into the image.

    Arguments:
        patches (list[np.ndarray]): List of patches.
        patch_size (int): Patch size in pixels.
        conv_offset (int): Convolutional offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels
        src_shape (tuple[H, W]): Source image shape.

    Returns:
        image (Tensor): Combined image.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    image = torch.zeros(
        patches[0].shape[:2] + src_shape, dtype=torch.float, requires_grad=True
    ).cuda()
    density = torch.zeros(*src_shape, dtype=torch.float).cuda()
    coords = _get_patch_coords(src_shape, patch_size, conv_offset, overlap)
    for i, coord in enumerate(coords):
        y, x = coord
        y0, y1 = y + conv_offset, y + patch_size - conv_offset
        x0, x1 = x + conv_offset, x + patch_size - conv_offset
        image[..., y0:y1, x0:x1] += patches[i][
            ...,
            conv_offset : patch_size - conv_offset,
            conv_offset : patch_size - conv_offset,
        ]
        density[..., y0:y1, x0:x1] += 1
    density[density == 0] = 1
    image /= density
    return image
