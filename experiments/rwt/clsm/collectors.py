"""Public-action collection policies; no hidden regime feedback."""
from __future__ import annotations

import numpy as np


class SmoothActionCollector:
    """Smoke-only bounded excitation policy, independent of evaluator law."""

    policy_id = "smooth_random_smoke_v1"

    def __init__(self, low: np.ndarray, high: np.ndarray, seed: int):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.rng = np.random.default_rng(seed)
        self.previous = np.zeros_like(self.low)

    def act(self, observation: np.ndarray) -> np.ndarray:
        if not np.isfinite(observation).all():
            raise ValueError("nonfinite public observation")
        self.previous = np.clip(.85 * self.previous + .15 * self.rng.normal(size=len(self.low)), self.low, self.high)
        return self.previous.copy()
