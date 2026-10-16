"""Stage 3A.1 CPU diagnostics: train-only ridge baseline and adapter memory.

No API in this module accepts a regime label, schedule, or physics parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rwt.stage3a import LinearResidualAdapter


@dataclass(frozen=True)
class TrainNormalizer:
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    delta_mean: np.ndarray
    delta_scale: np.ndarray

    @classmethod
    def fit(cls, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        delta = next_state - state
        values = (state, action, delta)
        means = [np.asarray(x, np.float64).mean(0) for x in values]
        scales = [np.asarray(x, np.float64).std(0).clip(1e-4) for x in values]
        return cls(means[0], scales[0], means[1], scales[1], means[2], scales[2])

    def normalized_mse(self, prediction: np.ndarray, next_state: np.ndarray) -> float:
        return float(np.mean(((prediction - next_state) / self.delta_scale) ** 2))


class LinearRidgeDynamics:
    """Closed-form delta predictor using A/B/D train statistics only."""

    def __init__(self, normalizer: TrainNormalizer, ridge: float = 1.0):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.normalizer = normalizer
        self.ridge = ridge
        self.weights: np.ndarray | None = None

    def _features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        norm = self.normalizer
        return np.concatenate((np.ones((len(state), 1)),
                               (state - norm.state_mean) / norm.state_scale,
                               (action - norm.action_mean) / norm.action_scale), axis=1)

    def fit(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        x = self._features(state, action)
        y = (next_state - state - self.normalizer.delta_mean) / self.normalizer.delta_scale
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("nonfinite ridge weights")
        return self

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit the ridge model before prediction")
        state = np.asarray(state, np.float64)
        action = np.asarray(action, np.float64)
        if state.ndim == 1:
            return self.predict(state[None], action[None])[0]
        delta = (self._features(state, action) @ self.weights) * self.normalizer.delta_scale
        return state + delta + self.normalizer.delta_mean


@dataclass
class MemoryEntry:
    adapter: LinearResidualAdapter
    signature: np.ndarray


class ResidualAdapterMemory:
    """Persist and retrieve observed residual adapters by predictive fit.

    A caller supplies only past observed transitions for retrieval. Probe inputs
    are fixed from training data and contain no target outputs or regime labels.
    """

    def __init__(self, probe_state: np.ndarray, probe_action: np.ndarray):
        self.probe_state = np.asarray(probe_state, np.float64)
        self.probe_action = np.asarray(probe_action, np.float64)
        if (self.probe_state.ndim != 2 or self.probe_action.ndim != 2
                or len(self.probe_state) != len(self.probe_action)):
            raise ValueError("probe arrays must have aligned rank-2 shapes")
        self.entries: list[MemoryEntry] = []

    def add(self, adapter: LinearResidualAdapter) -> int:
        if adapter.updates < 1:
            raise ValueError("cannot cache an untrained adapter")
        signature = np.stack([adapter.predict(s, a) for s, a in zip(self.probe_state, self.probe_action)])
        cloned = LinearResidualAdapter(adapter.state_dim, adapter.action_dim, ridge=1.0,
                                       state_mean=adapter.state_mean, state_scale=adapter.state_scale,
                                       action_mean=adapter.action_mean, action_scale=adapter.action_scale)
        cloned.weights[:] = adapter.weights
        cloned.covariance[:] = adapter.covariance
        cloned.updates = adapter.updates
        self.entries.append(MemoryEntry(cloned, signature))
        return len(self.entries) - 1

    def retrieve(self, observed_state: np.ndarray, observed_action: np.ndarray,
                 observed_next_state: np.ndarray,
                 base_predict: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> tuple[int, np.ndarray]:
        if not self.entries:
            raise ValueError("memory is empty")
        state = np.asarray(observed_state, np.float64)
        action = np.asarray(observed_action, np.float64)
        next_state = np.asarray(observed_next_state, np.float64)
        if (state.ndim != 2 or action.ndim != 2 or next_state.shape != state.shape
                or len(state) == 0 or len(state) != len(action)):
            raise ValueError("retrieval needs aligned observed transitions")
        base = np.asarray(base_predict(state, action), np.float64)
        errors = np.asarray([
            np.mean((base + np.stack([entry.adapter.predict(s, a) for s, a in zip(state, action)])
                     - next_state) ** 2)
            for entry in self.entries
        ])
        return int(np.argmin(errors)), errors

    def save(self, path: str | Path) -> None:
        if not self.entries:
            raise ValueError("cannot save empty memory")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adapters = [item.adapter for item in self.entries]
        np.savez_compressed(path, probe_state=self.probe_state, probe_action=self.probe_action,
                            weights=np.stack([x.weights for x in adapters]),
                            covariance=np.stack([x.covariance for x in adapters]),
                            state_mean=np.stack([x.state_mean for x in adapters]),
                            state_scale=np.stack([x.state_scale for x in adapters]),
                            action_mean=np.stack([x.action_mean for x in adapters]),
                            action_scale=np.stack([x.action_scale for x in adapters]),
                            updates=np.asarray([x.updates for x in adapters], np.int64),
                            signatures=np.stack([item.signature for item in self.entries]))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            memory = cls(data["probe_state"], data["probe_action"])
            for i in range(len(data["updates"])):
                adapter = LinearResidualAdapter(data["state_mean"].shape[1],
                                                data["action_mean"].shape[1], ridge=1.0,
                                                state_mean=data["state_mean"][i],
                                                state_scale=data["state_scale"][i],
                                                action_mean=data["action_mean"][i],
                                                action_scale=data["action_scale"][i])
                adapter.weights[:] = data["weights"][i]
                adapter.covariance[:] = data["covariance"][i]
                adapter.updates = int(data["updates"][i])
                signature = data["signatures"][i].copy()
                expected = np.stack([adapter.predict(s, a) for s, a in zip(memory.probe_state, memory.probe_action)])
                if not np.allclose(signature, expected, atol=1e-9):
                    raise ValueError("stored predictive signature does not match adapter")
                memory.entries.append(MemoryEntry(adapter, signature))
        return memory
