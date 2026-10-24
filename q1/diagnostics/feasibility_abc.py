"""Reusable, split-aware utilities for Q1 feasibility diagnostics A/B/C."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from q1.diagnostics.u4_crossfit import effect_and_prediction_metrics
from q1.simulator import DynamicsParams, PROGRAMS, apply_program, sample_params


def parameter_vector(params: DynamicsParams) -> np.ndarray:
    return np.asarray([*params.B.reshape(-1), params.d, params.b, params.kappa], dtype=np.float64)


def params_from_vector(theta: Iterable[float]) -> DynamicsParams:
    x = np.asarray(theta, dtype=np.float64)
    if x.shape != (7,):
        raise ValueError("theta must contain B[2x2], d, b, kappa")
    return DynamicsParams(x[:4].reshape(2, 2), float(x[4]), float(x[5]), float(x[6]))


def vector_response(params: DynamicsParams, probes: np.ndarray, *, dt: float) -> np.ndarray:
    """Vectorized physical conditional-mean delta-v response on [q,v,a]."""
    x = np.asarray(probes, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError("probes must be finite [n,6] state-action rows")
    q, v, actions = x[:, :2], x[:, 2:4], x[:, 4:]
    force = (np.tanh(actions @ params.B.T) - params.d * v
             - params.b * np.linalg.norm(v, axis=1, keepdims=True) * v - params.kappa * q)
    return float(dt) * force


def estimate_old_params(states: np.ndarray, actions: np.ndarray, delta_v: np.ndarray, *,
                        dt: float, seed: int, max_nfev: int = 400,
                        initializations: int = 8) -> tuple[DynamicsParams, dict]:
    """Multistart bounded equation-aware fit using old-stage transitions only.

    The API deliberately has no post-change observations or parameter labels.
    Bounds are broad generic simulator-feasible ranges, not region-specific.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    target = np.asarray(delta_v, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4 or actions.shape != (len(states), 2):
        raise ValueError("old states/actions have incompatible shapes")
    if target.shape != (len(states), 2) or not np.isfinite(np.c_[states, actions, target]).all():
        raise ValueError("old transition data must be finite and aligned")
    q, v = states[:, :2], states[:, 2:]
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    lo = np.asarray([-1.5, -1.5, -1.5, -1.5, 0.01, 0.01, 0.01])
    hi = np.asarray([1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])

    def residual(theta: np.ndarray) -> np.ndarray:
        bmat = theta[:4].reshape(2, 2)
        pred = float(dt) * (np.tanh(actions @ bmat.T) - theta[4] * v
                            - theta[5] * speed * v - theta[6] * q)
        return (pred - target).reshape(-1)

    # Predeclared neutral starts and deterministic random starts; never seeded from truth.
    starts = [np.asarray([s, 0.0, 0.0, s, d, b, k])
              for s, d, b, k in ((0.65, .25, .15, .2), (1.0, .5, .5, .5),
                                  (.35, .9, .25, .8), (1.25, .15, .8, .1))]
    rng = np.random.default_rng(int(seed))
    while len(starts) < max(1, int(initializations)):
        starts.append(np.asarray([rng.uniform(.25, 1.25), rng.uniform(-.1, .1),
                                  rng.uniform(-.1, .1), rng.uniform(.25, 1.25),
                                  rng.uniform(.05, .9), rng.uniform(.03, .9),
                                  rng.uniform(.03, .9)]))
    fits = []
    for x0 in starts[:max(1, int(initializations))]:
        result = least_squares(residual, np.clip(x0, lo + 1e-8, hi - 1e-8),
                               bounds=(lo, hi), method="trf", x_scale="jac",
                               max_nfev=int(max_nfev), ftol=1e-10, xtol=1e-10, gtol=1e-10)
        singular = np.linalg.svd(result.jac, compute_uv=False)
        tol = np.finfo(float).eps * max(result.jac.shape) * (singular[0] if singular.size else 0)
        rank = int(np.sum(singular > tol))
        fits.append({"x": result.x, "cost": float(result.cost), "success": bool(result.success),
                     "status": int(result.status), "message": str(result.message),
                     "nfev": int(result.nfev), "optimality": float(result.optimality),
                     "jacobian_singular_values": singular.tolist(), "jacobian_rank": rank,
                     "jacobian_condition_number": (float(singular[0] / singular[-1])
                         if len(singular) and singular[-1] > tol else None)})
    best_index = min(range(len(fits)), key=lambda i: (fits[i]["cost"], not fits[i]["success"]))
    best = fits[best_index]
    theta_hat = params_from_vector(best["x"])
    pred = vector_response(theta_hat, np.c_[states, actions], dt=dt)
    residuals = pred - target
    result_meta = {k: v for k, v in best.items() if k != "x"}
    result_meta.update({"selected_initialization": best_index,
                        "starts": [{k: v for k, v in fit.items() if k != "x"} for fit in fits],
                        "theta_hat": parameter_vector(theta_hat).tolist(),
                        "old_support_mse": float(np.mean(residuals ** 2)),
                        "old_support_rmse": float(np.sqrt(np.mean(residuals ** 2))),
                        "bounds": {"B_entries": [-1.5, 1.5], "d_b_kappa": [.01, 2.0]},
                        "fit_data_scope": "old-stage transitions only"})
    return theta_hat, result_meta


def functional_class_ids(params: DynamicsParams, *, dt: float, tolerance: float = 1e-7,
                         seed: int = 9182, probe_count: int = 256) -> tuple[list[int], np.ndarray]:
    """Simulator-only class partition from fixed probes, distinct from raw program IDs."""
    rng = np.random.default_rng(seed)
    probes = np.c_[rng.uniform(-.7, .7, (probe_count, 2)),
                   rng.uniform(-.5, .5, (probe_count, 2)),
                   rng.uniform(-1., 1., (probe_count, 2))]
    signatures = np.stack([vector_response(apply_program(params, p), probes, dt=dt)
                           for p in PROGRAMS])
    classes: list[int] = []
    representatives: list[np.ndarray] = []
    for sig in signatures:
        match = next((idx for idx, prior in enumerate(representatives)
                      if np.max(np.abs(sig - prior)) <= tolerance), None)
        if match is None:
            match = len(representatives)
            representatives.append(sig)
        classes.append(match)
    distances = np.sqrt(np.mean((signatures[:, None] - signatures[None, :]) ** 2, axis=(2, 3)))
    return classes, distances


def score_candidates(theta: DynamicsParams, probes: np.ndarray, observed_delta: np.ndarray,
                     *, dt: float) -> np.ndarray:
    """Physical MSE candidate scores in PROGRAMS order."""
    return np.asarray([np.mean((vector_response(apply_program(theta, p), probes, dt=dt)
                                - observed_delta) ** 2) for p in PROGRAMS], dtype=np.float64)


def old_transition_statistics(states: np.ndarray, actions: np.ndarray,
                              observed_delta: np.ndarray) -> np.ndarray:
    """Compact old-data summary for the learned transition-statistic B baseline."""
    s = np.asarray(states, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    dv = np.asarray(observed_delta, dtype=np.float64)
    v, q = s[:, 2:4], s[:, :2]
    speed = np.linalg.norm(v, axis=1)
    features = [np.mean(s, axis=0), np.std(s, axis=0), np.mean(a, axis=0),
                np.std(a, axis=0), np.mean(dv, axis=0), np.std(dv, axis=0),
                np.asarray([np.mean(speed), *np.quantile(speed, [.1, .5, .9]),
                            np.mean(speed ** 2), np.mean(speed ** 3)]),
                np.linalg.eigvalsh(np.cov(a, rowvar=False)),
                np.linalg.eigvalsh(np.cov(v, rowvar=False)),
                np.mean(q * v, axis=0), np.mean(speed[:, None] * v, axis=0)]
    return np.concatenate(features).astype(np.float64)


def fit_ridge_probe(train_x: np.ndarray, train_y: np.ndarray, inner_x: np.ndarray,
                    inner_y: np.ndarray, ridge_grid: Iterable[float]) -> tuple[dict, dict]:
    """Standardized multivariate ridge selected only on inner training backgrounds."""
    x, y = np.asarray(train_x, float), np.asarray(train_y, float)
    xv, yv = np.asarray(inner_x, float), np.asarray(inner_y, float)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-8)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-8)
    xn, yn = (x - xm) / xs, (y - ym) / ys
    xvn, yvn = (xv - xm) / xs, (yv - ym) / ys
    losses = []
    def solve(xx: np.ndarray, yy: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
        mu_x, mu_y = xx.mean(0), yy.mean(0)
        xc, yc = xx - mu_x, yy - mu_y
        mat = xc.T @ xc + float(ridge) * np.eye(xc.shape[1])
        coef = np.linalg.lstsq(mat, xc.T @ yc, rcond=None)[0]
        return coef, mu_y - mu_x @ coef
    for ridge in ridge_grid:
        coef, intercept = solve(xn, yn, float(ridge))
        losses.append({"ridge": float(ridge), "inner_mse_standardized": float(np.mean((xvn @ coef + intercept - yvn) ** 2))})
    chosen = min(losses, key=lambda r: (r["inner_mse_standardized"], r["ridge"]))
    coef, intercept = solve(xn, yn, chosen["ridge"])
    fitted = {"x_mean": xm, "x_scale": xs, "y_mean": ym, "y_scale": ys,
              "coef": coef, "intercept": intercept, "ridge": chosen["ridge"]}
    meta = {"selected_ridge": chosen["ridge"], "inner_candidates": losses,
            "selection_split": "background-level inner split from original train backgrounds"}
    return fitted, meta


def predict_ridge_probe(model: dict, x: np.ndarray) -> np.ndarray:
    return (((np.asarray(x, float) - model["x_mean"]) / model["x_scale"]) @ model["coef"]
            + model["intercept"]) * model["y_scale"] + model["y_mean"]


def fit_ridge_fixed(train_x: np.ndarray, train_y: np.ndarray, ridge: float) -> dict:
    """Refit all declared training backgrounds using a previously selected ridge value."""
    x, y = np.asarray(train_x, float), np.asarray(train_y, float)
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-8)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-8)
    xn, yn = (x-xm)/xs, (y-ym)/ys
    mx, my = xn.mean(0), yn.mean(0)
    xc, yc = xn-mx, yn-my
    coef = np.linalg.lstsq(xc.T@xc + float(ridge)*np.eye(xc.shape[1]), xc.T@yc, rcond=None)[0]
    return {"x_mean": xm, "x_scale": xs, "y_mean": ym, "y_scale": ys,
            "coef": coef, "intercept": my-mx@coef, "ridge": float(ridge)}


def effect_metrics(true_old: np.ndarray, true_new: np.ndarray, pred_old: np.ndarray,
                   pred_effect: np.ndarray) -> dict:
    return effect_and_prediction_metrics(pred_old=pred_old, pred_new=pred_old + pred_effect,
                                         true_old=true_old, true_new=true_new)


def mean_ci_by_background(values: Iterable[float], z: float = 1.96) -> list[float] | None:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return None
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return [mean - float(z) * se, mean + float(z) * se]


def sample_idlike_manifest(*, seed: int, count: int, train_ranges: dict,
                           forbidden_parameter_seeds: Iterable[int]) -> dict:
    """Draw new train-range parameter seeds while avoiding all saved environment seeds."""
    forbidden = {int(x) for x in forbidden_parameter_seeds}
    rng = np.random.default_rng(int(seed))
    selected = []
    while len(selected) < int(count):
        candidate = int(rng.integers(1, 2**31 - 1))
        if candidate not in forbidden and candidate not in selected:
            selected.append(candidate)
    backgrounds = [sample_params(np.random.default_rng(s), "train", train_ranges) for s in selected]
    return {"regime": "IDLIKE_DEVELOPMENT_ONLY", "seed": int(seed),
            "parameter_region": "train", "environment_ids": [f"idlike-dev-{i:04d}" for i in range(count)],
            "parameter_seeds": selected, "forbidden_saved_parameter_seeds_checked": sorted(forbidden),
            "backgrounds": [{"B": p.B.tolist(), "d": p.d, "b": p.b, "kappa": p.kappa} for p in backgrounds]}


def split_training_background_indices(*, count: int, seed: int, fit_count: int,
                                      validation_count: int) -> tuple[list[int], list[int]]:
    """Reproducible disjoint fit/inner-validation partition over train backgrounds only."""
    if fit_count < 1 or validation_count < 1 or fit_count + validation_count > count:
        raise ValueError("fit and validation partitions must be nonempty and within train count")
    permutation=np.random.default_rng(int(seed)).permutation(int(count))
    val=sorted(int(i) for i in permutation[:validation_count])
    fit=sorted(int(i) for i in permutation[validation_count:validation_count+fit_count])
    if set(fit)&set(val):
        raise AssertionError("training-background partitions overlap")
    return fit,val


def summarize_backgrounds(rows: list[dict], *, keys: tuple[str, ...], value_key: str,
                          z: float = 1.96) -> dict:
    """Cluster summaries by background; caller must average repeated seeds within background first."""
    result = {}
    for key_tuple in sorted({tuple(row[k] for k in keys) for row in rows}, key=str):
        subset = [r for r in rows if tuple(r[k] for k in keys) == key_tuple]
        vals = np.asarray([r[value_key] for r in subset if r[value_key] is not None], dtype=float)
        label = "/".join(str(x) for x in key_tuple)
        result[label] = {"n_backgrounds": int(len(vals)), "mean": float(vals.mean()) if len(vals) else None,
                         "ci95": mean_ci_by_background(vals, z)}
    return result
