"""Opaque-code block diagonal affine coefficient operators."""
from __future__ import annotations
import torch
from torch import nn


class AffineOperatorBank(nn.Module):
    def __init__(self, widths: tuple[int, ...] = (6, 4, 2), codes: int = 4):
        super().__init__()
        self.widths = widths
        self.blocks = nn.ParameterList([
            nn.Parameter(torch.eye(w).repeat(codes, 1, 1)) for w in widths
        ])
        self.bias = nn.Parameter(torch.zeros(codes, sum(widths)))

    def matrix(self, code: int) -> torch.Tensor:
        rows = []
        for i, block in enumerate(self.blocks):
            row = [torch.zeros(block.shape[-1], other.shape[-1], dtype=block.dtype,
                               device=block.device) for other in self.blocks]
            row[i] = block[code]
            rows.append(torch.cat(row, dim=1))
        return torch.cat(rows, dim=0)

    def apply(self, code: int, mean: torch.Tensor, covariance: torch.Tensor | None = None):
        a = self.matrix(code)
        b = self.bias[code]
        out_mean = a @ mean + b
        if covariance is None:
            return out_mean
        return out_mean, a @ covariance @ a.T

    @staticmethod
    def compose(left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]):
        """Compose Ui then Uj: T_j o T_i."""
        ai, bi = left
        aj, bj = right
        return aj @ ai, aj @ bi + bj

    @staticmethod
    def identity(k: int, *, dtype=None, device=None):
        return torch.eye(k, dtype=dtype, device=device), torch.zeros(k, dtype=dtype, device=device)

    def regularization(self) -> torch.Tensor:
        value = self.bias.square().mean()
        for block in self.blocks:
            eye = torch.eye(block.shape[-1], dtype=block.dtype, device=block.device)
            value = value + (block - eye).square().mean()
        return value
