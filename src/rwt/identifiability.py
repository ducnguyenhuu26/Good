"""Diagnostics and temporal inference helpers for RWT theory identifiability."""

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from torch import Tensor

from rwt.dynamics import DynamicsParameters


def composition_closure(
    regimes: dict[str, DynamicsParameters], state: np.ndarray, action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ground-truth finite-difference closure and its relative scale.

    ``residual = F_C - F_B - F_D + F_A`` is diagnostic-only. The relative
    denominator is the sum of the two observed single-mechanism changes from A.
    """
    required = {"A", "B", "C", "D"}
    if set(regimes) != required:
        raise ValueError(f"regimes must be exactly {sorted(required)}")
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.shape[-1] != 4 or action.shape[-1] != 2 or state.shape[:-1] != action.shape[:-1]:
        raise ValueError("state/action probes must have matching leading dimensions and shapes (...,4)/(...,2)")
    flat_s, flat_a = state.reshape(-1, 4), action.reshape(-1, 2)
    predictions = {}
    for name, parameters in regimes.items():
        predictions[name] = np.stack([parameters.predict_state(s, a) for s, a in zip(flat_s, flat_a)]).reshape(state.shape)
    fa, fb, fc, fd = (predictions[name] for name in ("A", "B", "C", "D"))
    residual = fc - fb - fd + fa
    denominator = np.linalg.norm(fb - fa, axis=-1) + np.linalg.norm(fd - fa, axis=-1)
    relative = np.linalg.norm(residual, axis=-1) / np.maximum(denominator, 1e-12)
    return residual, relative


def additive_prediction(base: np.ndarray, primitive_a: np.ndarray, primitive_b: np.ndarray) -> np.ndarray:
    """Evaluate the held-out 11 code of a two-factor additive theory."""
    return np.asarray(base) + np.asarray(primitive_a) + np.asarray(primitive_b)


def fit_three_code_assignment(
    observations: dict[str, np.ndarray], assignment: dict[str, tuple[int, int]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit additive code values to three observed regimes under a proposed map.

    This tiny algebraic construction is used to demonstrate permutation/anchor
    non-identifiability, not as a dynamics learner.
    """
    required_regimes = {"A", "B", "D"}
    required_codes = {(0, 0), (1, 0), (0, 1)}
    if set(observations) != required_regimes or set(assignment) != required_regimes:
        raise ValueError("exactly A/B/D observations and assignments are required")
    if set(assignment.values()) != required_codes:
        raise ValueError("A/B/D must map bijectively to codes 00, 10, 01")
    by_code = {code: np.asarray(observations[name]) for name, code in assignment.items()}
    base = by_code[(0, 0)]
    p1 = by_code[(1, 0)] - base
    p2 = by_code[(0, 1)] - base
    reconstructed = {
        name: base + code[0] * p1 + code[1] * p2
        for name, code in assignment.items()
    }
    return reconstructed, additive_prediction(base, p1, p2)


def hmm_log_evidence(log_likelihood: Tensor, hazard: float) -> Tensor:
    """Forward algorithm log evidence for uniform-start persistent latent codes.

    ``log_likelihood`` has shape ``[time, codes]``. The transition matrix is
    ``T=(1-h)I + h/K`` so each transition has total change probability ``h``
    (including the possibility of redrawing the same code).
    """
    if log_likelihood.ndim != 2 or log_likelihood.shape[0] < 1 or log_likelihood.shape[1] < 1:
        raise ValueError("log_likelihood must have shape (nonempty_time, nonempty_codes)")
    if not 0.0 <= hazard <= 1.0:
        raise ValueError("hazard must be in [0,1]")
    time, count = log_likelihood.shape
    transition = torch.full((count, count), hazard / count, dtype=log_likelihood.dtype, device=log_likelihood.device)
    transition.diagonal().add_(1.0 - hazard)
    log_transition = transition.log()
    alpha = log_likelihood[0] - np.log(count)
    for t in range(1, time):
        alpha = torch.logsumexp(alpha[:, None] + log_transition, dim=0) + log_likelihood[t]
    return torch.logsumexp(alpha, dim=0)


def hmm_filter_step(posterior: np.ndarray, log_likelihood: np.ndarray, hazard: float) -> np.ndarray:
    """One causal HMM predict/update step for NumPy evaluation code."""
    posterior = np.asarray(posterior, np.float64)
    likelihood = np.asarray(log_likelihood, np.float64)
    if posterior.ndim != 1 or likelihood.shape != posterior.shape:
        raise ValueError("posterior and likelihood must be equal-length vectors")
    if not 0.0 <= hazard <= 1.0 or np.any(posterior < 0) or posterior.sum() <= 0:
        raise ValueError("invalid HMM posterior or hazard")
    prior = (1.0 - hazard) * posterior + hazard / len(posterior)
    logp = np.log(prior.clip(1e-300)) + likelihood
    logp -= logp.max()
    result = np.exp(logp)
    return result / result.sum()


def enumerate_hmm_log_evidence(log_likelihood: Tensor, hazard: float) -> Tensor:
    """Slow exact enumeration used only by tiny correctness tests."""
    time, count = log_likelihood.shape
    trans = torch.full((count, count), hazard / count, dtype=log_likelihood.dtype, device=log_likelihood.device)
    trans.diagonal().add_(1.0 - hazard)
    terms = []
    for path in product(range(count), repeat=time):
        value = log_likelihood.new_tensor(-np.log(count)) + log_likelihood[0, path[0]]
        for t in range(1, time):
            value = value + trans[path[t - 1], path[t]].log() + log_likelihood[t, path[t]]
        terms.append(value)
    return torch.logsumexp(torch.stack(terms), dim=0)


__all__ = [
    "additive_prediction", "composition_closure", "enumerate_hmm_log_evidence",
    "fit_three_code_assignment", "hmm_filter_step", "hmm_log_evidence",
]
