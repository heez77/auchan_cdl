import warnings

import torch
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

def multi_category_focal_loss2(y_true, y_pred):
    alpha = .25
    gamma = 2.
    epsilon = 1.e-7

    y_true = torch.Tensor(y_true)
    y_pred = torch.clamp(y_pred, epsilon, 1.-epsilon)
    y_pred = torch.Tensor(y_pred)

    alpha_t = y_true*alpha + (torch.ones_like(y_true)-y_true)*(1-alpha)
    y_t = torch.multiply(y_true, y_pred) + torch.multiply(1-y_true, 1-y_pred)
    ce = -torch.log(y_t)
    weight = torch.pow(torch.subtract(torch.Tensor(1), y_t), gamma)
    print(weight)
    fl = torch.multiply(torch.multiply(weight, ce), alpha_t)
    loss = torch.mean(fl)
    return loss

class CustomLoss(_Loss):
    def __init__(self) -> None:
        super(CustomLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return multi_category_focal_loss2(input, target)