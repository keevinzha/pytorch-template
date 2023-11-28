# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/23 18:25
@Auth ： keevinzha
@File ：losses.py
@IDE ：PyCharm
"""

import torch

from util.recon_tools import ifft1c

def mae_loss(pred, target, niter):
    target = ifft1c(target, 2)
    pred_length = 4
    cost_all = 0
    for i in range(niter):
        pred_per_iter = ifft1c(pred[i], 2)
        error = torch.abs(target - pred_per_iter)
        cost = torch.mean(torch.sum(error, dim=tuple(range(len(pred[0].shape)))[1:]), dim=0)
        cost_all += cost
    cost_mean = cost_all / niter
    return cost_mean

class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, target, niter):
        return mae_loss(pred, target, niter)
