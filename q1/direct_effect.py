"""Small diagnostic direct function-space effect networks for Q1 experiment C."""
from __future__ import annotations

import torch
from torch import nn


class DirectEffectMLP(nn.Module):
    """Predict normalized physical delta-v from normalized x, old context, opaque code."""

    def __init__(self, context_dim: int, *, hidden_dim: int = 64,
                 embedding_dim: int = 8, n_codes: int = 4, use_context: bool = True):
        super().__init__()
        self.context_dim = int(context_dim)
        self.use_context = bool(use_context)
        self.code_embedding = nn.Embedding(n_codes, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(6 + context_dim + embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x_norm: torch.Tensor, context_norm: torch.Tensor,
                opaque_code: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2 or x_norm.shape[-1] != 6:
            raise ValueError("x_norm must have shape [n,6]")
        if context_norm.ndim != 2 or context_norm.shape != (len(x_norm), self.context_dim):
            raise ValueError("context_norm must have shape [n,context_dim]")
        code = opaque_code.reshape(-1).to(device=x_norm.device, dtype=torch.long)
        if code.shape != (len(x_norm),):
            raise ValueError("opaque_code must contain one code per input row")
        context = context_norm if self.use_context else torch.zeros_like(context_norm)
        embedding = self.code_embedding(code)
        return self.network(torch.cat((x_norm, context, embedding), dim=-1))


def inference_effect(model: DirectEffectMLP, x_norm: torch.Tensor,
                     context_norm: torch.Tensor, opaque_code: torch.Tensor,
                     code_scales: torch.Tensor) -> torch.Tensor:
    """Inference API intentionally accepts no simulator params/program/labels."""
    model.eval()
    with torch.no_grad():
        normalized = model(x_norm, context_norm, opaque_code)
        scales = code_scales.to(device=normalized.device, dtype=normalized.dtype)[opaque_code.reshape(-1)]
        if scales.ndim == 1:
            scales = scales.unsqueeze(-1)
        return normalized * scales
