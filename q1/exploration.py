"""Fixed, reproducible exploration policy shared by all Q1 methods."""

from __future__ import annotations

import numpy as np


def action_at(t: int, rng: np.random.Generator, limit: float = 1.0,
              signal_scale: float = 0.85, jitter_scale: float = 0.15) -> np.ndarray:
    """Bounded noisy sinusoidal action; uses no simulator or intervention metadata."""
    phase = 0.37 * t
    signal = np.array([np.sin(phase), np.cos(phase * 0.73)])
    return np.clip(signal_scale * signal + rng.uniform(-jitter_scale, jitter_scale, 2), -limit, limit)


def action_sequence(n: int, seed: int, limit: float = 1.0,
                    signal_scale: float = 0.85, jitter_scale: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([action_at(t, rng, limit, signal_scale, jitter_scale) for t in range(n)])
