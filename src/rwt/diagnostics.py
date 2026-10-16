"""Pairwise predictive KL and action-discrimination diagnostics."""

from __future__ import annotations

from itertools import combinations
import numpy as np
from numpy.typing import NDArray

from rwt.dynamics import DynamicsParameters
from rwt.agents import probe_action_set


FloatArray = NDArray[np.float64]


def transition_kl(
    left: DynamicsParameters,
    right: DynamicsParameters,
    state: FloatArray,
    action: FloatArray,
    noise_std: float,
) -> float:
    """KL[N(mu_left,sigma²I) || N(mu_right,sigma²I)] for next velocity."""

    sigma = max(float(noise_std), 1e-6)
    delta = left.predict_velocity(state[2:], action) - right.predict_velocity(state[2:], action)
    return float(np.square(delta).sum() / (2.0 * sigma**2))


def pairwise_kl_matrix(
    candidates: tuple[DynamicsParameters, ...],
    state: FloatArray,
    action: FloatArray,
    noise_std: float,
) -> FloatArray:
    count = len(candidates)
    result = np.zeros((count, count), dtype=np.float64)
    means = np.stack([candidate.predict_velocity(state[2:], action) for candidate in candidates])
    squared = np.square(means[:, None, :] - means[None, :, :]).sum(axis=2)
    result[:] = squared / (2.0 * max(float(noise_std), 1e-6) ** 2)
    return result


def candidate_pair_rows(
    candidates: tuple[DynamicsParameters, ...],
    *,
    noise_std: float,
    max_action_norm: float,
) -> list[dict[str, object]]:
    """Long-form KL table for all oracle candidate pairs at diagnostic states."""

    states = {
        "rest": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "moving": np.asarray([0.0, 0.0, 1.0, -0.5]),
    }
    axes = max_action_norm * np.asarray([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                                          [2**-0.5, 2**-0.5], [2**-0.5, -2**-0.5]])
    actions = {f"probe_{index}": value for index, value in enumerate(axes)}
    rows: list[dict[str, object]] = []
    for state_name, state in states.items():
        for action_name, action in actions.items():
            for i, j in combinations(range(len(candidates)), 2):
                rows.append({
                    "state_case": state_name,
                    "action_case": action_name,
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                    "candidate_i": i,
                    "candidate_j": j,
                    "kl_nats": transition_kl(candidates[i], candidates[j], state, action, noise_std),
                })
    return rows


def discrimination_summary(
    truth: DynamicsParameters,
    candidates: tuple[DynamicsParameters, ...],
    state: FloatArray,
    action: FloatArray,
    noise_std: float,
) -> tuple[float, float]:
    """Mean and minimum KL from current truth to alternative library models."""

    divergences = [
        transition_kl(truth, candidate, state, action, noise_std)
        for candidate in candidates
        if candidate != truth
    ]
    return float(np.mean(divergences)), float(np.min(divergences))


def best_probe_actions(
    candidates: tuple[DynamicsParameters, ...],
    state: FloatArray,
    *,
    noise_std: float,
    max_action_norm: float,
) -> list[tuple[FloatArray, float, float]]:
    """Rank shared probe actions by mean and weakest-pair KL separation."""

    ranked = []
    for action in probe_action_set(max_action_norm):
        upper = pairwise_kl_matrix(candidates, state, action, noise_std)[np.triu_indices(len(candidates), 1)]
        ranked.append((action.copy(), float(np.mean(upper)), float(np.min(upper))))
    return sorted(ranked, key=lambda row: (row[1], row[2]), reverse=True)
