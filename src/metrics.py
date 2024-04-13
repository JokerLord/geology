import torch
from torch import Tensor

from collections import namedtuple


def iou(y_true: Tensor, y_pred: Tensor, smooth: float = 1.0) -> float:
    """
    Arguments:
        y_true (Tensor): Ground truth tensor of size (H, W)
        y_pred (Tensor): Prediction tensor of size (H, W)
        smooth (float, Optional): Smooth coefficient. Default: 1.0

    Returns:
        iou (float): Calculated IoU metric
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth).item() / (union + smooth).item()


def iou_per_class(y_true: Tensor, y_pred: Tensor, smooth: float = 1.0) -> list[float]:
    """
    Arguments:
        y_true (Tensor): One hot encoded tensor of size (n_classes, H, W)
        y_pred (Tensor): Prediction tensor of size (n_classes, H, W)
        smooth (float, Optional): Smooth coefficient. Default: 1.0

    Returns:
        iou_vals (list[IoU]): List of IoU metrics per each of n_classes classes
    """
    iou_vals = []
    for i in range(y_true.shape[0]):
        iou_vals.append(iou(y_true[i], y_pred[i], smooth))
    return iou_vals
