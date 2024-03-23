import numpy as np
import math


def _get_patch_coords(img_shape, patch_size, conv_offset, overlap):
    """
    Arguments:
        img_shape (tuple): Image shape.
        patch_size (int): Patch size in pixels.
        conv_offset (int): Convolutional offset in pixels.
        overlap (int): Overlap number in pixels.

    Returns:
        coords (list[(tuple)]): List of patch coordinates.
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


def split_into_patches(image, patch_size, conv_offset, overlap):
    """
    Splits image into patches.

    Arguments:
        image (np.ndarray): Source image.
        patch_size (int): Patch size in pixels.
        conv_offset (int): Convolutional offset in pixels.
        overlap (int or float): either float in [0, 1] (fraction of patch size)
            or int in pixels

    Returns:
        patchs (list[np.ndarray]): list of extracted patches.
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
