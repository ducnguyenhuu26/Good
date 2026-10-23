"""Exhaustive learned-program inference over the fixed 21-program vocabulary."""

from __future__ import annotations

import torch

from q1.simulator import PROGRAMS


@torch.no_grad()
def rank_programs(context: torch.Tensor, adaptation: torch.Tensor, *, encoder, executor,
                  primitives, normalizer, length_penalty: float = 0.0,
                  predictive_variance: float = 1e-4,
                  device: torch.device | None = None) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """Rank using only an old context and the supplied new-stage transitions."""
    if adaptation.ndim != 2 or adaptation.shape[1] != 8:
        raise ValueError("adaptation must be an (N, 8) public transition array")
    device = device or context.device
    x = torch.as_tensor(adaptation[:, :6], dtype=torch.float32, device=device)
    y = torch.as_tensor(adaptation[:, 6:], dtype=torch.float32, device=device)
    xn, yn = normalizer.normalize_x(x), normalizer.normalize_dv(y)
    target = normalizer.denormalize_dv(yn)
    transformed_contexts = torch.stack([
        primitives.apply_program(context, program).squeeze(0) for program in PROGRAMS
    ])
    x_batch = xn.unsqueeze(0).expand(len(PROGRAMS), -1, -1)
    prediction = normalizer.denormalize_dv(executor(x_batch, transformed_contexts))
    target_batch = target.unsqueeze(0)
    variance = torch.as_tensor(2.0 * torch.pi * predictive_variance,
                               dtype=prediction.dtype, device=device)
    nll = 0.5 * (torch.log(variance) + (prediction - target_batch) ** 2 / predictive_variance)
    lengths = torch.as_tensor([len(p) for p in PROGRAMS], dtype=prediction.dtype, device=device)
    values = torch.sum(nll, dim=(1, 2)) + length_penalty * lengths
    order = torch.argsort(values).cpu().tolist()
    return [PROGRAMS[i] for i in order], values[order]
