import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProb(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(LinearProb, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.linear_head = nn.Conv2d(dim, n_classes, (1, 1))

    def forward(self, x):
        x_detached = torch.clone(x.detach())
        linear_head_probs = self.linear_head(x_detached)
        return linear_head_probs