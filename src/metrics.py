import torch

from torch import Tensor
from config import *
from collections import namedtuple


exIoU = namedtuple("IoU", ["iou", "intersection", "union"])
exAcc = namedtuple("Accuracy", ["accuracy", "correct", "total"])


def iou(y_true: Tensor, y_pred: Tensor, smooth: float = 1.0) -> exIoU:
    """
    Arguments:
        y_true (Tensor): Ground truth tensor of size (H, W)
        y_pred (Tensor): Prediction tensor of size (H, W)
        smooth (float, Optional): Smooth coefficient. Default: 1.0
    Returns:
        res (exIoU): Calculated extended IoU metric
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return exIoU(
        (intersection + smooth).item() / (union + smooth).item(),
        intersection.item(),
        union.item()
    )


def iou_per_class(
    y_true: Tensor, y_pred: Tensor, smooth: float = 1.0
) -> dict[str, exIoU]:
    """
    Arguments:
        y_true (Tensor): One hot encoded tensor of size (n_classes, H, W)
        y_pred (Tensor): Prediction tensor of size (n_classes, H, W)
        smooth (float, Optional): Smooth coefficient. Default: 1.0
    Returns:
        iou_dict (dict[str, exIoU]): Dictionary of extended IoU metrics for each class
    """
    iou_dict = {}
    for i in range(y_true.shape[0]):
        iou_dict[squeezed_codes2labels[i]] = iou(y_true[i], y_pred[i], smooth)
    return iou_dict


def joint_iou(ious: list[exIoU], smooth: float = 1.0) -> exIoU:
    intersection = sum(i.intersection for i in ious)
    union = sum(i.union for i in ious)
    return exIoU(
        (intersection + smooth) / (union + smooth),
        intersection,
        union
    )


def accuracy(y_true: Tensor, y_pred: Tensor) -> exAcc:
    """
    Arguments:
        y_true (Tensor): One hot encoded tensor of size (n_classes, H, W)
        y_pred (Tensor): Prediction tensor of size (n_classes, H, W)
    Returns:
        accuracy (exAcc): Extended accuracy metric
    """
    y_true_a = torch.argmax(y_true, dim=0)
    y_pred_a = torch.argmax(y_pred, dim=0)
    correct = torch.sum(y_true_a == y_pred_a)
    return exAcc(
        correct.item() / torch.numel(y_true_a),
        correct,
        torch.numel(y_true_a)
    )

def joint_accuracy(accs: list[exAcc]) -> exAcc:
    correct = sum(a.correct for a in accs)
    total = sum(a.total for a in accs)
    return exAcc(correct / total, correct, total)
