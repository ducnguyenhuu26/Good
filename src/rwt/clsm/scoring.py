"""Proper finite-sample law-forecast scoring."""
from __future__ import annotations

import torch
from torch import Tensor


def energy_score(samples: Tensor, target: Tensor) -> Tensor:
    """Unbiased pairwise estimate for iid signature draws [M,D]."""
    if samples.ndim != 2 or len(samples) < 2 or target.shape != (samples.shape[1],):
        raise ValueError("energy score requires M>=2 samples [M,D] and target [D]")
    m = len(samples)
    first = torch.linalg.vector_norm(samples - target, dim=1).mean()
    pairwise = torch.pdist(samples, p=2).sum() / (m * (m - 1))
    return first - pairwise
