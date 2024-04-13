import torch
from torch import Tensor

from collections import namedtuple


iou_extended = namedtuple("IoU", ["iou", "intersection", "union"])


def iou(y_true: Tensor, y_pred: Tensor, smooth=1.0) -> iou_extended:
    """
    Arguments:
        y_true (Tensor): One hot encoded tensor of size (n_classes, H, W)
        y_pred (Tensor): Prediction tensor of size (n_classes, H, W)
        smooth (float, Optional): Smooth coefficient. Default: 1.0

    Returns:
        iou_extended (IoU): Calculated IoU metric with intersection and
            union for joint iou calculation
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou_extended(iou, intersection, union)
