import numpy as np
import math
import torch

from typing import Union
from torch import Tensor


def _get_patch_coords(img_shape, patch_size, offset, overlap):
    """
    Arguments:
        img_shape (tuple[int, int, int, int]): Image shape.
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int): Overlap number in pixels.

    Returns:
        coords (list[(int, int)]): List of patch coordinates.
    """

    H, W = img_shape[-2:]
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
        image (Tensor): Source image.
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels

    Returns:
        patchs (list[np.ndarray]): List of extracted patches.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    coords = _get_patch_coords(image.shape, patch_size, offset, overlap)
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
) -> Tensor:
    """
    Combines patches back into the image.

    Arguments:
        patches (list[np.ndarray]): List of patches.
        patch_size (int): Patch size in pixels.
        offset (int): Offset in pixels.
        overlap (int or float): Either float in [0, 1] (fraction of patch size)
            or int in pixels
        src_shape (tuple[H, W]): Source image shape.

    Returns:
        image (Tensor): Combined image.
    """

    if isinstance(overlap, float):
        overlap = int(patch_size * overlap)
    image = torch.zeros(patches[0].shape[:1] + src_shape, dtype=torch.float, requires_grad=True).cuda()
    density = torch.zeros(*src_shape, dtype=torch.float).cuda()
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


def one_hot(mask, n_classes):
    """
    Applies one hot encoding

    Arguments:
        mask (np.ndarray): Input tensor with size (x.H, x.W).
        n_classes (int): Total number of classes.

    Return:
        new_mask (np.ndarray): Output tensor with size (n_classes, x.H, x.W)
    """

    new_mask = []
    for i in range(1, n_classes + 1):
        new_mask.append(mask == i)
    new_mask = np.array(new_mask, np.uint8)
    return new_mask.transpose([1, 2, 0])
