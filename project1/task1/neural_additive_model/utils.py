# This file includes the utility functions for the neural additive model:

# -------------------------------
# Packages and Presets
# -------------------------------
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.parameter import Parameter

# -------------------------------
# Custom Weight Initialization
# -------------------------------
#!!! This function was copied from the following post in the pytorch forum:
# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
def truncated_normal(x: torch.Tensor, mean: float=4.0, std: float=0.5) -> torch.Tensor:
    size = x.shape
    tmp = x.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    x.data.mul_(std).add_(mean)
    return x

# -------------------------------
# Custom Activation Functions
# -------------------------------

class ExU(nn.Module):
    def __init__(self):
        super(ExU, self).__init__()
        # the original paper used a truncated normal with mean 4 and sd 0.5 
        # for ExU initialization.
        # see: https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py

    def forward(self, x: torch.Tensor, weight: Parameter, bias: Parameter) -> torch.Tensor:
        return torch.exp(weight) @ (x - bias)
    

class ReLUn(nn.Module):
    def __init__(self, n: float=1.0):
        super(ReLUn, self).__init__()
        # the original paper used 'glorot_uniform" for relu initialization
        # see: https://github.com/google-research/google-research/blob/master/neural_additive_models/models.py
        self.n = n
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.clip(F.relu(x), min=0, max=self.n)