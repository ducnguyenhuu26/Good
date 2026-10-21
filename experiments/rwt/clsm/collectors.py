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


class NominalControllerMixture:
    """Predeclared public-observation collection mixture for full data.

    The nominal policy object must expose `predict(obs, deterministic=True)`.
    Its provenance is recorded separately by the dataset builder. No hidden
    law, switch, physics array, or oracle field enters this policy.
    """

    policy_id = "nominal_sac_smooth_excitation_mixture_v1"

    def __init__(self, low: np.ndarray, high: np.ndarray, policy, seed: int):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.smooth = np.zeros_like(self.low)
        self.phase = self.rng.uniform(0, 2 * np.pi, size=len(self.low))
        self.time = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        if not np.isfinite(observation).all():
            raise ValueError("nonfinite public observation")
        mode = int(self.rng.integers(0, 3))
        if mode == 0:
            action, _ = self.policy.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float64) + self.rng.normal(0, .05, len(self.low))
        elif mode == 1:
            self.smooth = .85 * self.smooth + .15 * self.rng.normal(size=len(self.low))
            action = self.smooth
        else:
            action = .55 * np.sin(self.phase + .13 * self.time)
            action += .1 * self.rng.normal(size=len(self.low))
        self.time += 1
        return np.clip(action, self.low, self.high)
