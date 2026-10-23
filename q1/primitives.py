"""Learned latent transformation primitives; no simulator imports here."""

from __future__ import annotations

import torch
from torch import nn


class PrimitiveBank(nn.Module):
    def __init__(self, context_dim: int = 32, embedding_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(4, embedding_dim) * 0.05)
        self.residual = nn.Sequential(
            nn.Linear(context_dim + embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def apply_one(self, context: torch.Tensor, code: int) -> torch.Tensor:
        if code not in range(4):
            raise ValueError("primitive code must be in [0, 3]")
        embedding = self.embeddings[code].expand(*context.shape[:-1], -1)
        return context + self.residual(torch.cat((context, embedding), dim=-1))

    def apply_program(self, context: torch.Tensor, program: tuple[int, ...]) -> torch.Tensor:
        result = context
        for code in program:
            result = self.apply_one(result, code)
        return result

    def forward(self, context: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Batched one-code application, used by M3 optimization."""
        emb = self.embeddings[codes]
        return context + self.residual(torch.cat((context, emb), dim=-1))


def map_intervention_program(program: tuple[int, ...], group_to_code: dict[str, int]) -> tuple[int, ...]:
    return tuple(group_to_code[str(code)] for code in program)


class DirectUpdateBank(nn.Module):
    """Unstructured per-code residual update baseline for M3 ablations."""
    def __init__(self, context_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, context_dim))
            for _ in range(4)
        ])

    def apply_one(self, context: torch.Tensor, code: int) -> torch.Tensor:
        if code not in range(4):
            raise ValueError("primitive code must be in [0, 3]")
        return context + self.updates[code](context)

    def forward(self, context: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(context)
        for code, update in enumerate(self.updates):
            mask = codes == code
            if torch.any(mask):
                result[mask] = context[mask] + update(context[mask])
        return result
