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


def boundary_distance_BCE(inp, target, weights):
    # Smoothing needed to avoid divide by 0 error
    smooth = 1.0
    flattened_inp = inp.view(-1)
    flattened_target = target.view(-1)
    flattened_weights = weights.view(-1)
    criterion = nn.BCELoss(reduce=False)
    loss = criterion(flattened_inp, flattened_target)
    loss = (loss * flattened_weights).mean()
    return loss


def boundary_distance_dice(inp, target, weights):
    # Smoothing needed to avoid divide by 0 error
    smooth = 1.0
    flattened_inp = inp.view(-1)
    flattened_target = target.view(-1)
    flattened_weights = weights.view(-1)
    flattened_inp = flattened_inp * flattened_weights
    flattened_target = flattened_target * flattened_weights
    common = 2 * torch.sum(flattened_target * flattened_inp) + smooth
    total = torch.sum(flattened_target) + torch.sum(flattened_inp) + smooth
    return - common / total


class DistToBoundary(object):
    def __init__(self, targets):
        self.targets = targets
        self.shape = self.targets[0].shape

        self.all_targets_maps = []
        for target in self.targets:
            cur_target_map = []
            for k in range(4):
                _, rows, cols = target.shape
                row_boundary = []
                col_boundary = []
                for i in range(rows):
                    row_boundary.append([])
                    row_boundary[i].append(0)
                    cur_val = target[k, i, 0]

                    for j in range(1, cols-1):
                        if cur_val != target[k, i, j]:
                            row_boundary[i].append(j)
                            cur_val = target[k, i, j]

                    row_boundary[i].append(cols-1)

                for j in range(cols):
                    col_boundary.append([])
                    col_boundary[j].append(0)
                    cur_val = target[k, 0, j]

                    for i in range(1, rows-1):
                        if cur_val != target[k, i, j]:
                            col_boundary[j].append(i)
                            cur_val = target[k, i, j]

                    col_boundary[j].append(rows-1)
                cur_target_map.append((row_boundary, col_boundary))
            self.all_targets_maps.append(cur_target_map)

    def getClosest(self, ind):
        """
        Get a of weights for closest boundary distance
        """
        channels, rows, cols = self.shape
        row_target = list(range(cols))
        col_target = list(range(rows))
        row_weight = np.zeros((channels, rows, cols))
        col_weight = np.zeros((channels, rows, cols))

        for k in range(channels):
            rmap, cmap = self.all_targets_maps[ind][k]

            for i in range(rows):
                tmp_sum = np.reshape(row_target, (-1, 1)) - \
                                     np.reshape(rmap[i], (1, -1))
                row_wt = np.min(np.abs(tmp_sum), axis=1)
                row_weight[k, i, :] = row_wt

            for j in range(cols):
                tmp_sum = np.reshape(col_target, (-1, 1)) - \
                                     np.reshape(cmap[j], (1, -1))
                col_wt = np.min(np.abs(tmp_sum), axis=1)
                col_weight[k, :, j] = col_wt

        return row_weight, col_weight

    def computeWeights(self):
        row_weights = []
        col_weights = []
        for i in range(len(self.targets)):
            r, c = self.getClosest(i)
            row_weights.append(r)
            col_weights.append(c)
        return np.array(row_weights), np.array(col_weights)


def lovasz_hinge_loss(inp, target):
    """
        See paper, Algorithm 1 in page 4
        https://arxiv.org/abs/1705.08790
    """
    flattened_inp = inp.view(-1)
    flattened_target = target.view(-1)

    y_val = 2 * (flattened_target - 0.5)
    m_val = nn.functional.relu(1. - flattened_inp * flattened_target)

    sorted_m, indices = torch.sort(m_val, dim=0, descending=True,
                                   output=None)
    sorted_y = labels[indices]

    # Calculate g_i (from Algorithm 1)
    p = sorted_y.size()
    sum_delta = sorted_y.sum()
    intersection = sum_delta - sorted_y.cumsum(0)
    union = sum_delta + (1 - sorted_y).cumsum(0)
    g_val = 1 - intersection / union
    g_val[1:p] = g_val[1:p] - g_val[0:p-1]
    loss = (g_val * sorted_m).sum()

    return loss
