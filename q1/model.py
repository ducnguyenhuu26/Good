"""Learned context encoder and shared dynamics executor (M2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TransitionNormalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    dv_mean: np.ndarray
    dv_std: np.ndarray

    @classmethod
    def fit(cls, training_transitions: list[np.ndarray]) -> "TransitionNormalizer":
        values = np.concatenate(training_transitions, axis=0)
        x = values[:, :6]
        dv = values[:, 6:]
        x_std = x.std(axis=0)
        dv_std = dv.std(axis=0)
        return cls(x.mean(axis=0), np.maximum(x_std, 1e-5), dv.mean(axis=0), np.maximum(dv_std, 1e-5))

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - torch.as_tensor(self.x_mean, dtype=x.dtype, device=x.device)) / torch.as_tensor(
            self.x_std, dtype=x.dtype, device=x.device
        )

    def normalize_dv(self, dv: torch.Tensor) -> torch.Tensor:
        return (dv - torch.as_tensor(self.dv_mean, dtype=dv.dtype, device=dv.device)) / torch.as_tensor(
            self.dv_std, dtype=dv.dtype, device=dv.device
        )

    def denormalize_dv(self, dv: torch.Tensor) -> torch.Tensor:
        return dv * torch.as_tensor(self.dv_std, dtype=dv.dtype, device=dv.device) + torch.as_tensor(
            self.dv_mean, dtype=dv.dtype, device=dv.device
        )

    def as_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "dv_mean", "dv_std")}

    @classmethod
    def from_dict(cls, obj: dict) -> "TransitionNormalizer":
        return cls(*(np.asarray(obj[k], dtype=np.float32) for k in ("x_mean", "x_std", "dv_mean", "dv_std")))


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int,
         dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ContextEncoder(nn.Module):
    def __init__(self, context_dim: int = 32, hidden_dim: int = 128, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.transition_net = _mlp(8, hidden_dim, hidden_dim, depth, dropout)
        self.context_net = _mlp(hidden_dim, context_dim, hidden_dim, 1, dropout)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        if support.ndim != 3 or support.shape[-1] != 8:
            raise ValueError("support must have shape (batch, transitions, 8)")
        return self.context_net(self.transition_net(support).mean(dim=1))


class DynamicsExecutor(nn.Module):
    def __init__(self, context_dim: int = 32, hidden_dim: int = 128, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(6 + context_dim, 2, hidden_dim, depth, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 6 or context.shape[-1] + 6 != self.net[0].in_features:
            raise ValueError("executor expects x[...,6] and configured context vectors")
        if context.ndim == 2 and x.ndim == 3:
            context = context[:, None, :].expand(-1, x.shape[1], -1)
        elif context.ndim == 2 and x.ndim == 2 and context.shape[0] == 1 and x.shape[0] > 1:
            context = context.expand(x.shape[0], -1)
        return self.net(torch.cat((x, context), dim=-1))


class ContextDynamicsModel(nn.Module):
    def __init__(self, context_dim: int = 32, hidden_dim: int = 128, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = ContextEncoder(context_dim, hidden_dim, depth, dropout)
        self.executor = DynamicsExecutor(context_dim, hidden_dim, depth, dropout)

    def encode(self, support: torch.Tensor) -> torch.Tensor:
        return self.encoder(support)

    def predict(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.executor(x, context)

    def forward(self, support: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x, self.encode(support))


class ContextFreeDynamicsModel(nn.Module):
    """Matched executor architecture with a fixed zero context."""
    def __init__(self, context_dim: int = 32, hidden_dim: int = 128, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.executor = DynamicsExecutor(context_dim, hidden_dim, depth, dropout)
        self.register_buffer("zero_context", torch.zeros(1, context_dim))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        context = self.zero_context.expand(x.shape[0], -1)
        return self.executor(x, context)
