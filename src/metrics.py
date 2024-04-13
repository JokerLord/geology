import torch
from torch import Tensor


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
        iou_vals (list[float]): List of IoU metrics per each of n_classes classes
    """
    iou_vals = []
    for i in range(y_true.shape[0]):
        iou_vals.append(iou(y_true[i], y_pred[i], smooth))
    return iou_vals


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Arguments:
        y_true (Tensor): One hot encoded tensor of size (n_classes, H, W)
        y_pred (Tensor): Prediction tensor of size (n_classes, H, W)

    Returns:
        accuracy (float): Accuracy metric
    """
    y_true_a = torch.argmax(y_true, dim=0)
    y_pred_a = torch.argmax(y_pred, dim=0)
    correct = torch.sum(y_true_a == y_pred_a)
    return correct.item() / torch.numel(y_true_a)
