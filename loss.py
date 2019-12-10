import torch 
import torch.nn as nn
import numpy as np


def dice(img1, img2):
    """
    Implemented in numpy to check correcteness
    """
    # Smoothing needed to avoid divide by 0 error
    smooth_val = 1.0
    img1 = np.array(img1, dtype=np.bool)
    img2 = np.array(img2, dtype=np.bool)
    common = 2 * np.sum(np.multiply(img1, img2)) + smooth_val
    return (np.sum(common)) / (np.sum(img1) + np.sum(img2) + smooth_val)


def dice_pytorch(inp, target):
    # Smoothing needed to avoid divide by 0 error
    smooth = 1.0
    flattened_inp = inp.view(-1)
    flattened_target = target.view(-1)
    common = 2 * torch.sum(flattened_target * flattened_inp) + smooth
    total = torch.sum(flattened_target) + torch.sum(flattened_inp) + smooth
    return -common / total


def dice_BCE(inp, target):
    smooth = 1.0

    flattened_inp = inp.view(-1)
    flattened_target = target.view(-1)
    common = 2 * torch.sum(flattened_target * flattened_inp) + smooth
    total = torch.sum(flattened_target) + torch.sum(flattened_inp) + smooth
    loss1 = - common / total

    criterion2 = nn.BCELoss()
    loss2 = criterion2(inp, target)

    alpha = 2.0
    return alpha * loss1 + loss2
