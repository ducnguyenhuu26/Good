"""Bounded pre-change exploration and stage-aware local information utilities.

The action schedule is exogenous: it is a function of time and a seed only.
No simulator parameter, intervention code, or future observation enters policy
construction or action selection.
"""
from __future__ import annotations

import numpy as np

from .active_design import full_response_jacobian
from .identifiability import efficient_information


ENERGY_TARGET = 1.8
POLICIES = (
    "ORIGINAL_SINUSOIDAL",
    "BLOCK_FULL_24",
    "BLOCK_24_ENERGY_MATCHED",
    "RANDOM_CORNER_ENERGY_MATCHED",
    "MULTIFREQ_SMOOTH_ENERGY_MATCHED",
    "BLOCK_ROTATED_ENERGY_MATCHED",
    "BLOCK_DWELL_12_ENERGY_MATCHED",
    "BLOCK_DWELL_48_ENERGY_MATCHED",
)


def functional_class(program: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalize commuting program orders for evaluation only."""
    p = tuple(program)
    if len(p) == 2 and set(p) == {0, 1}:
        return p
    return tuple(sorted(p))


def action_sequence(policy: str, *, seed: int, n: int = 96,
                    energy_target: float = ENERGY_TARGET) -> np.ndarray:
    """Build a bounded open-loop action sequence using only policy, seed, time."""
    from q1.exploration import action_at

    if n < 1 or not 0 < energy_target < 2:
        raise ValueError("n must be positive and energy_target must lie in (0,2)")
    rng = np.random.default_rng(seed)
    if policy == "ORIGINAL_SINUSOIDAL":
        actions = np.asarray([action_at(t, rng) for t in range(n)], dtype=np.float64)
    elif policy in {"BLOCK_FULL_24", "BLOCK_24_ENERGY_MATCHED",
                    "BLOCK_ROTATED_ENERGY_MATCHED", "BLOCK_DWELL_12_ENERGY_MATCHED",
                    "BLOCK_DWELL_48_ENERGY_MATCHED"}:
        patterns = np.asarray([[1., -1.], [1., 1.], [-1., 1.], [-1., -1.]])
        dwell = 24
        if "DWELL_12" in policy:
            dwell = 12
        elif "DWELL_48" in policy:
            dwell = 48
        if "ROTATED" in policy:
            patterns = patterns[[0, 2, 1, 3]]
        amplitude = 1.0 if policy == "BLOCK_FULL_24" else np.sqrt(energy_target / 2.0)
        actions = np.asarray([patterns[(t // dwell) % 4] * amplitude for t in range(n)])
    elif policy == "RANDOM_CORNER_ENERGY_MATCHED":
        signs = rng.choice(np.asarray([-1., 1.]), size=(n, 2))
        actions = signs * np.sqrt(energy_target / 2.0)
    elif policy == "MULTIFREQ_SMOOTH_ENERGY_MATCHED":
        t = np.arange(n, dtype=np.float64) / n
        phase = rng.uniform(-np.pi, np.pi, size=4)
        raw = np.column_stack((
            np.sin(2*np.pi*2*t + phase[0]) + .55*np.sin(2*np.pi*5*t + phase[1]),
            np.cos(2*np.pi*3*t + phase[2]) + .45*np.sin(2*np.pi*7*t + phase[3]),
        ))
        lo, hi = 0., 128.
        for _ in range(80):
            mid = .5 * (lo + hi)
            e = float(np.mean(np.sum(np.clip(mid * raw, -1., 1.) ** 2, axis=1)))
            if e < energy_target:
                lo = mid
            else:
                hi = mid
        actions = np.clip(.5*(lo+hi)*raw, -1., 1.)
        # A finite sampled waveform can miss the target at its saturation limit.
        if abs(float(np.mean(np.sum(actions**2, axis=1))) - energy_target) > 1e-8:
            raise RuntimeError("smooth control could not attain the declared action energy")
    else:
        raise ValueError(f"unknown pre-change policy: {policy}")
    if actions.shape != (n, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.+1e-12:
        raise RuntimeError("action sequence violated shape, finiteness, or bound")
    return actions


def action_energy(actions: np.ndarray) -> dict[str, float]:
    a = np.asarray(actions, dtype=np.float64)
    norms2 = np.sum(a*a, axis=1)
    return {
        "sum_squared_norm": float(norms2.sum()),
        "mean_squared_norm": float(norms2.mean()),
        "rms_norm": float(np.sqrt(norms2.mean())),
        "max_abs_component": float(np.max(np.abs(a))),
    }


def estimate_multiplicative_ratio(feature: np.ndarray, residual: np.ndarray) -> float:
    """Estimate r in residual=r*feature+noise when the baseline has b=0."""
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    e = np.asarray(residual, dtype=np.float64).reshape(-1)
    if f.shape != e.shape or not np.isfinite(f).all() or not np.isfinite(e).all():
        raise ValueError("feature and residual must be finite vectors with equal shape")
    den = float(f @ f)
    if den <= np.finfo(float).tiny:
        raise ValueError("operator magnitude is unidentified on a zero-energy feature")
    ratio = float(f @ e) / den
    if not np.isfinite(ratio) or ratio <= 0:
        raise FloatingPointError("estimated multiplicative operator must be positive")
    return ratio


def matched_population_b(train_b: np.ndarray) -> dict[str, float | int]:
    """Linear-scale empirical Bayes calibration from training old-law fits."""
    b = np.asarray(train_b, dtype=np.float64)
    if b.ndim != 1 or len(b) < 2 or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("training b estimates must be a finite positive vector")
    return {"b_mean": float(b.mean()), "b_sd": float(b.std(ddof=1)), "n_train": int(len(b))}


def matched_population_u4_effect(x: np.ndarray, theta_reference: np.ndarray,
                                 response_fn, transform_fn, opaque_u4: int,
                                 training_calibration: dict[str, float | int]) -> np.ndarray:
    """Population U4 effect using the same learned operator and train-only E[b]."""
    theta = np.asarray(theta_reference, dtype=np.float64).copy()
    theta[5] = np.log(float(training_calibration["b_mean"]))
    after = transform_fn(theta, (int(opaque_u4),))
    return response_fn(x, after) - response_fn(x, theta)


def stage_corrected_jacobian(x_old: np.ndarray, x_post: np.ndarray,
                             theta_old: np.ndarray, program: tuple[int, ...],
                             model) -> np.ndarray:
    """Jacobian for (theta_old, shared response) with the post-stage chain rule.

    This is an offline candidate-design/evaluation utility. An online pre-change
    policy must not pass a true intervention identity; it should use old-stage
    information or average this quantity over a learned program posterior.
    """
    theta_post = model.transform(theta_old, program)
    j0 = full_response_jacobian(np.asarray(x_old), theta_old, model.c)
    j1 = full_response_jacobian(np.asarray(x_post), theta_post, model.c)
    m = model.transform_jac(program)
    post = np.column_stack((j1[:, :7] @ m, j1[:, 7:]))
    return np.vstack((j0, post))


def prechange_old_stage_information(x_old: np.ndarray, theta_old: np.ndarray, c,
                                    *, sigma: float) -> dict[str, float | int]:
    """Stage-correct old-only efficient information, independent of program."""
    jac = full_response_jacobian(np.asarray(x_old), theta_old, c)
    return efficient_information(jac, 5, sigma=sigma)
