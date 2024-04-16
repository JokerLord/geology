import numpy as np
import math
import torch
import matplotlib.pyplot as plt

from typing import Union
from torch import Tensor
from pathlib import Path
from config import *


present_class_codes = [code for code in range(len(CLASS_NAMES)) if code not in MISSED_CLASS_CODES]
codes2squeezed_codes = {code: i for i, code in enumerate(present_class_codes)}
squeezed_codes2labels = {i: CLASS_NAMES[code] for i, code in enumerate(present_class_codes)}
labels2colors = {class_name: CLASS_COLORS[i] for i, class_name in enumerate(CLASS_NAMES)}


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


def mean_iou(iou_per_class: dict[str, float], weights=None) -> float:
    """
    Calculates average IoU metric over all classes

    Arguments:
        iou_per_class (dict[str, float]): Dictionary of IoU metrics for each class
        weights (list, Optional): List of weights for each class. Default: None
    
    Returns:
        mean_iou (float): Weighted mean iou
    """

    if weights is None:
        return sum(iou for iou in iou_per_class.values()) / len(iou_per_class)
    else:
        """ Not implemented yet """
        pass


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


def plot_single_class_data(data: list, data_name: str, exp_path: Path):
    n_epochs = len(data)
    fig = plt.figure(figsize=(12, 6))
    
    x = [x + 1 for x in range(n_epochs)]
    y = [data[i] for i in range(n_epochs)]
    plt.plot(x, y)

    plt.ylabel(f"{data_name}", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    fig.savefig(exp_path / f"{data_name}.png")


def plot_multi_class_data(data: dict[str, list[float]], data_name: str, exp_path: Path):
    n_epochs = len(list(data.values())[0])
    fig = plt.figure(figsize=(12, 6))

    for class_name, values in data.items():
        x = [x + 1 for x in range(n_epochs)]
        y = [values[i] for i in range(n_epochs)]
        plt.plot(x, y, color=labels2colors[class_name])

    plt.ylabel(f"{data_name}", fontsize=20)
    plt.xlabel("epoch", fontsize=20)    
    plt.legend([class_name for class_name in data], loc="center right", fontsize=15)
    fig.savefig(exp_path / f"{data_name}.png")


def save_training_outputs(epoch_outputs: list[dict], exp_path: Path) -> None:
    """
    Creates plots and metrics output into exp_path folder

    Arguments:
        epoch_outputs (list[dict]): List of dictionaries with epoch average metrics over val_dataloader 
    """
    # description = "val"
    # output_path = self.exp_path / description
    # output_path.mkdir(exist_ok=True, parents=True)

    # log_file = open(output_path / "metrics.txt", "a+")
    # data = {} # dict[str, list[float]]
    # avg_data = {} # dict[str, float]
    # for key in epoch_outputs[0].keys():
    #     if key in ["iou_activated_per_class", "iou_pred_per_class"]:
    #         data[key] = {class_name: [x[key][class_name] for x in epoch_outputs] for class_name in epoch_outputs[0][key].keys()}
    #         plot_multi_class_data(data[key], key, exp_path)
    #     else:
    #         data[key] = [x[key] for x in epoch_outputs]
    #         plot_single_class_data(data[key], key, exp_path)

    # """ Write metrics by epoch """
    # for epoch in range(len(epoch_outputs)):
    #     write_data = epoch_outputs[epoch]
        
    data = {} # dict[str, list[float]]
    for key in epoch_outputs[0].keys():
        data[key] = [x[key] for x in epoch_outputs]
        plot_single_class_data(data[key], key, exp_path)


def metrics_to_str(data: dict[str, Union[float, str]], description: str) -> str:
    iou_activated = "".join(f"\t\t {class_name}: {iou:.4f}\n" for class_name, iou in data["iou_activated_per_class"].items())
    iou_pred = "".join(f"\t\t {class_name}: {iou:.4f}\n" for class_name, iou in data["iou_pred_per_class"].items())
    res_str = (
        f"Evaluation result ({description}):\n"
        f"\tmean IoU (activated): {data["mean_iou_activated"]:.4f}\n"
        f"\tmean IoU (prediction): {data["mean_iou_pred"]:.4f}\n"
        f"\taccuracy: {data["accuracy"]:.4f}\n"
        f"\tIoU (activated) per class:\n"
        f"{iou_activated}"
        f"\tIoU (prediction) per class:\n"
        f"{iou_pred}\n"
    )
    return res_str


def write_metrics(file: object, data: dict[str, Union[float, dict[str, float]]], description: str) -> None:
    file.write(metrics_to_str(data, description))
