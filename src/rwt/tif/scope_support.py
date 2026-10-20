"""Shared mathematical primitives used by the law architectures."""

from __future__ import annotations

import numpy as np

Batch = tuple[np.ndarray, np.ndarray, np.ndarray]


def hazard(age: int, scale: float, shape: float) -> float:
    """Discrete Weibull conditional switch probability given survival to age."""
    return float(-np.expm1(-(((age + 1) / scale) ** shape - (age / scale) ** shape)))
