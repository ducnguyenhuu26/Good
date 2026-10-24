"""Learned additive transition-function basis; contains no simulator knowledge."""
from __future__ import annotations

import torch
from torch import nn


class _PartitionBasis(nn.Module):
    def __init__(self, input_dim: int, width: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * width))
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(*x.shape[:-1], 2, self.width)


class PartitionedBasis(nn.Module):
    """Phi(x) in R^(2 x K), using independently learned action/v/q maps."""
    def __init__(self, widths: tuple[int, int, int] = (6, 4, 2), hidden: int = 32):
        super().__init__()
        if sum(widths) != 12:
            raise ValueError("v2 requires K=12 partitioned as (6,4,2)")
        self.widths = widths
        self.action = _PartitionBasis(2, widths[0], hidden)
        self.velocity = _PartitionBasis(2, widths[1], hidden)
        self.position = _PartitionBasis(2, widths[2], hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 6:
            raise ValueError("basis input must be [q_x,q_y,v_x,v_y,a_x,a_y]")
        return torch.cat((self.action(x[..., 4:6]), self.velocity(x[..., 2:4]),
                          self.position(x[..., :2])), dim=-1)


class UnrestrictedBasis(nn.Module):
    """Matched output shape ablation using unrestricted six-dimensional input."""
    def __init__(self, hidden: int = 32, k: int = 12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 2 * k))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(*x.shape[:-1], 2, self.k)
