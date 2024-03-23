import numpy as np


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
    return np.asarray(new_mask, dtype=float)