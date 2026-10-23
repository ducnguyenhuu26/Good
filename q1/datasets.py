"""Deterministic trajectory and split-manifest generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q1.exploration import action_at
from q1.simulator import DynamicsParams, apply_program, sample_params, transition


@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray
    actions: np.ndarray

    @property
    def delta_v(self) -> np.ndarray:
        return self.states[1:, 2:] - self.states[:-1, 2:]


def rollout(params: DynamicsParams, *, n: int, seed: int, dt: float = 0.05,
            sigma: float = 0.0, initial_state: np.ndarray | None = None,
            initial_q_std: float = 0.25, initial_v_std: float = 0.25,
            action_limit: float = 1.0, action_signal_scale: float = 0.85,
            action_jitter: float = 0.15) -> Trajectory:
    init_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 104729)
    noise_rng = np.random.default_rng(seed + 209759)
    init = (init_rng.normal(0.0, [initial_q_std, initial_q_std, initial_v_std, initial_v_std]) if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).copy())
    states = np.empty((n + 1, 4), dtype=np.float64)
    actions = np.empty((n, 2), dtype=np.float64)
    states[0] = init
    for t in range(n):
        actions[t] = action_at(t, action_rng, action_limit, action_signal_scale, action_jitter)
        states[t + 1] = transition(states[t], actions[t], params, dt=dt, sigma=sigma, rng=noise_rng)
        if np.linalg.norm(states[t + 1]) > 100:
            raise FloatingPointError(f"unstable trajectory at transition {t}")
    return Trajectory(states, actions)


def make_split_manifest(seed: int = 2026, counts: dict[str, int] | None = None) -> dict:
    counts = counts or {"train": 48, "validation": 16, "test_id": 24,
                        "test_background_ood": 24}
    rng = np.random.default_rng(seed)
    manifest = {"seed": seed, "regions": {}, "environment_ids": {}, "environment_seeds": {}}
    for split, count in counts.items():
        manifest["regions"][split] = split
        manifest["environment_ids"][split] = [f"{split}-{i:04d}" for i in range(count)]
        manifest["environment_seeds"][split] = [
            seed + sum((i + 1) * ord(c) for i, c in enumerate(f"{split}:{j}"))
            for j in range(count)
        ]
    ids = [x for values in manifest["environment_ids"].values() for x in values]
    assert len(ids) == len(set(ids))
    manifest["parameter_seed"] = int(rng.integers(0, 2**31 - 1))
    return manifest


def sample_environment(manifest: dict, split: str, index: int,
                       parameter_ranges: dict | None = None) -> DynamicsParams:
    seed = manifest["environment_seeds"][split][index]
    return sample_params(np.random.default_rng(seed), manifest["regions"][split], parameter_ranges)


def stage_pair(params: DynamicsParams, program: tuple[int, ...], *, seed: int,
               n_support: int = 80, n_query: int = 80, dt: float = 0.05,
               sigma: float = 0.0, initial_q_std: float = 0.25,
               initial_v_std: float = 0.25, action_limit: float = 1.0,
               action_signal_scale: float = 0.85, action_jitter: float = 0.15
               ) -> tuple[Trajectory, Trajectory, Trajectory]:
    """Old support, continuous adaptation, and independent after-stage query."""
    old = rollout(params, n=n_support, seed=seed, dt=dt, sigma=sigma,
                  initial_q_std=initial_q_std, initial_v_std=initial_v_std,
                  action_limit=action_limit, action_signal_scale=action_signal_scale,
                  action_jitter=action_jitter)
    after_params = apply_program(params, program)
    # One uninterrupted adaptation trajectory: no reset at the stage boundary.
    action_rng = np.random.default_rng(seed + 104730)
    noise_rng = np.random.default_rng(seed + 209760)
    states = np.empty((n_query + 1, 4), dtype=np.float64)
    actions = np.empty((n_query, 2), dtype=np.float64)
    states[0] = old.states[-1]
    for t in range(n_query):
        actions[t] = action_at(t, action_rng, action_limit, action_signal_scale, action_jitter)
        states[t + 1] = transition(states[t], actions[t], after_params, dt=dt, sigma=sigma, rng=noise_rng)
        if np.linalg.norm(states[t + 1]) > 100:
            raise FloatingPointError(f"unstable adaptation at transition {t}")
    adaptation = Trajectory(states, actions)
    query = rollout(after_params, n=n_query, seed=seed + 2, dt=dt, sigma=sigma,
                    initial_q_std=initial_q_std, initial_v_std=initial_v_std,
                    action_limit=action_limit, action_signal_scale=action_signal_scale,
                    action_jitter=action_jitter)
    return old, adaptation, query
