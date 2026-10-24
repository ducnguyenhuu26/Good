"""Cross-fitted U4 effect diagnostics with frozen M2 weights.

All effect and prediction errors are measured in physical delta-v units.
Functions that fit a displacement take fit labels only; query-label scoring is
intentionally kept in a separate API.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from q1.model import TransitionNormalizer


def ensure_write_once_directory(path: str | Path) -> Path:
    """Create a run directory, permitting only a previously-created empty directory."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty write-once output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def missing_required_files(paths: Iterable[str | Path]) -> list[str]:
    """Return absent required local artifacts without substituting or initializing them."""
    return [str(Path(path)) for path in paths if not Path(path).is_file()]


def _rows(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [n,2] array of physical delta-v values")
    return array


def mean_squared_error(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("MSE arrays must be nonempty and have identical shapes")
    return float(np.mean((left - right) ** 2))


def effect_and_prediction_metrics(
    *,
    pred_old: np.ndarray,
    pred_new: np.ndarray,
    true_old: np.ndarray,
    true_new: np.ndarray,
) -> dict:
    """Separate transformation fidelity from absolute after-law accuracy.

    Inputs and returned errors use raw physical delta-v units. No denominator
    floor is applied: effect skill is undefined for zero/nonfinite signal energy.
    """
    pred_old = _rows("pred_old", pred_old)
    pred_new = _rows("pred_new", pred_new)
    true_old = _rows("true_old", true_old)
    true_new = _rows("true_new", true_new)
    if not (pred_old.shape == pred_new.shape == true_old.shape == true_new.shape):
        raise ValueError("all prediction and truth arrays must have the same shape")
    true_effect = true_new - true_old
    pred_effect = pred_new - pred_old
    effect_energy = float(np.mean(true_effect**2))
    effect_mse = mean_squared_error(pred_effect, true_effect)
    if effect_energy > np.finfo(np.float64).tiny and np.isfinite(effect_energy):
        effect_skill = float(1.0 - effect_mse / effect_energy)
    else:
        effect_skill = None

    base_error = mean_squared_error(pred_old, true_old)
    persistence_error = mean_squared_error(pred_old, true_new)
    after_error = mean_squared_error(pred_new, true_new)
    perfect_after = pred_old + true_effect
    perfect_after_mse = mean_squared_error(perfect_after, true_new)
    perfect_gain = ((persistence_error - perfect_after_mse) / persistence_error
                    if persistence_error > 0 else None)
    after_gain = ((persistence_error - after_error) / persistence_error
                  if persistence_error > 0 else None)
    old_bias = pred_old - true_old
    effect_residual = pred_effect - true_effect
    decomposition_left = pred_new - true_new
    decomposition_right = old_bias + effect_residual
    decomposition_error = float(np.max(np.abs(decomposition_left - decomposition_right)))
    return {
        "probe_count": int(len(pred_old)),
        "effect_mse_delta_v_squared": effect_mse,
        "effect_energy_delta_v_squared": effect_energy,
        "effect_skill": effect_skill,
        "base_error_delta_v_squared": base_error,
        "persistence_error_delta_v_squared": persistence_error,
        "after_error_delta_v_squared": after_error,
        "perfect_effect_after_mse_delta_v_squared": perfect_after_mse,
        "perfect_effect_identity_abs_error": abs(perfect_after_mse - base_error),
        "perfect_effect_gain": perfect_gain,
        "learned_after_gain": after_gain,
        "correctable_gain_persistence_minus_base_delta_v_squared": persistence_error - base_error,
        "old_context_bias_mse_delta_v_squared": mean_squared_error(pred_old, true_old),
        "decomposition_max_abs_error_delta_v": decomposition_error,
        "decomposition_identity_pass": bool(np.allclose(
            decomposition_left, decomposition_right, rtol=1e-12, atol=1e-14)),
    }


def _svd_ridge_solve(matrix: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size or matrix.size == 0:
        raise ValueError("ridge system must be a nonempty matrix with matching target rows")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tol = (np.finfo(np.float64).eps * max(matrix.shape) * singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > tol))
    coeff = np.zeros_like(singular)
    active = singular > tol
    coeff[active] = singular[active] / (singular[active] ** 2 + ridge)
    displacement = vh.T @ (coeff * (u.T @ target))
    condition = (float(singular[0] / singular[rank - 1])
                 if rank == matrix.shape[1] and rank > 0 else None)
    return displacement, {
        "singular_values": singular.tolist(),
        "effective_rank": rank,
        "context_dimension": int(matrix.shape[1]),
        "rank_deficient": bool(rank < matrix.shape[1]),
        "condition_number": condition,
        "ridge_lambda": float(ridge),
        "solver": "numpy.linalg.svd ridge filter; no Gram inverse",
        "finite_displacement": bool(np.isfinite(displacement).all()),
    }


def select_fit_ridge_and_refit(
    jacobian_fit: np.ndarray,
    effect_fit: np.ndarray,
    jacobian_inner_train: np.ndarray,
    effect_inner_train: np.ndarray,
    jacobian_inner_validation: np.ndarray,
    effect_inner_validation: np.ndarray,
    *,
    ridge_factors: Iterable[float],
) -> tuple[np.ndarray, dict]:
    """Select ridge only on an inner fit split, then refit on all outer-fit data."""
    jt = np.asarray(jacobian_inner_train, dtype=np.float64)
    yt = np.asarray(effect_inner_train, dtype=np.float64).reshape(-1)
    jv = np.asarray(jacobian_inner_validation, dtype=np.float64)
    yv = np.asarray(effect_inner_validation, dtype=np.float64).reshape(-1)
    scale = float(np.mean(np.linalg.svd(jt, compute_uv=False) ** 2))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    candidates = []
    for factor in ridge_factors:
        lam = float(factor) * scale
        dc, _ = _svd_ridge_solve(jt, yt, lam)
        validation_mse = float(np.mean((jv @ dc - yv) ** 2))
        candidates.append({"ridge_factor": float(factor), "ridge_lambda": lam,
                           "inner_validation_effect_mse": validation_mse})
    if not candidates or not np.isfinite([c["inner_validation_effect_mse"] for c in candidates]).all():
        raise FloatingPointError("fit-only ridge selection produced no finite candidate")
    chosen = min(candidates, key=lambda item: (item["inner_validation_effect_mse"], item["ridge_factor"]))
    jacobian_fit = np.asarray(jacobian_fit, dtype=np.float64)
    effect_fit = np.asarray(effect_fit, dtype=np.float64).reshape(-1)
    displacement, solver = _svd_ridge_solve(jacobian_fit, effect_fit, chosen["ridge_lambda"])
    solver.update({"selected_ridge_factor": chosen["ridge_factor"],
                   "inner_fit_validation_candidates": candidates,
                   "selection_data": "inner fit labels only",
                   "alpha": 1.0})
    return displacement, solver


def predict_crossfit_query(model, normalizer: TransitionNormalizer, context_old: torch.Tensor,
                           displacement: np.ndarray, raw_probes: np.ndarray,
                           device: torch.device) -> dict:
    """Predict query effects from inputs/context only; deliberately accepts no query labels."""
    probes = np.asarray(raw_probes, dtype=np.float32)
    if probes.ndim != 2 or probes.shape[1] != 6 or not np.isfinite(probes).all():
        raise ValueError("raw_probes must be finite [n,6] state-action rows")
    x_raw = torch.as_tensor(probes, dtype=torch.float32, device=device)
    x_norm = normalizer.normalize_x(x_raw)
    c0 = context_old.reshape(1, -1)
    dc = torch.as_tensor(np.asarray(displacement, dtype=np.float32), device=device).reshape(1, -1)
    model.eval()
    with torch.no_grad():
        pred_old = normalizer.denormalize_dv(model.executor(x_norm, c0.expand(len(x_norm), -1)))
        pred_new = normalizer.denormalize_dv(model.executor(x_norm, (c0 + dc).expand(len(x_norm), -1)))
    return {"pred_old": pred_old.cpu().numpy().astype(np.float64),
            "pred_new": pred_new.cpu().numpy().astype(np.float64),
            "linear_effect": None}


def batched_context_jacobian(model, normalizer: TransitionNormalizer, context_old: torch.Tensor,
                             raw_probes: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-probe Jacobian d physical delta-v / d context, using input probes only."""
    probes = torch.as_tensor(np.asarray(raw_probes), dtype=torch.float32, device=device)
    if probes.ndim != 2 or probes.shape[1] != 6:
        raise ValueError("raw_probes must have shape [n,6]")
    x = normalizer.normalize_x(probes)
    context = context_old.reshape(-1).to(device=device, dtype=torch.float32)
    dv_std = torch.as_tensor(normalizer.dv_std, dtype=torch.float32, device=device)
    dv_mean = torch.as_tensor(normalizer.dv_mean, dtype=torch.float32, device=device)

    def physical_output(one_x: torch.Tensor, one_context: torch.Tensor) -> torch.Tensor:
        normalized = model.executor(one_x.unsqueeze(0), one_context.unsqueeze(0)).squeeze(0)
        return normalized * dv_std + dv_mean

    single_jacobian = torch.func.jacrev(physical_output, argnums=1)
    jac = torch.func.vmap(single_jacobian, in_dims=(0, None))(x, context)
    return jac.detach().cpu().numpy().astype(np.float64)


def solve_query_refit_upper_bound(jacobian_query: np.ndarray, effect_query: np.ndarray,
                                 ridge: float = 0.0) -> tuple[np.ndarray, dict]:
    """Explicitly leaky same-query-label projection; never use for feasibility selection."""
    dc, diagnostics = _svd_ridge_solve(jacobian_query, effect_query, ridge)
    diagnostics.update({"label": "query_refit_upper_bound",
                        "status": "QUERY-LABEL LEAKAGE / NONTRANSFER ORACLE",
                        "excluded_from_feasibility_gate": True})
    return dc, diagnostics


def blocked_inner_indices(n: int, validation_fraction: float = 0.25,
                          gap_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous early fit block and later inner-validation block separated by a gap."""
    if n < 16 or not 0 < validation_fraction < 0.5 or not 0 <= gap_fraction < 0.5:
        raise ValueError("invalid blocked inner-split size/fractions")
    n_val = max(4, int(round(n * validation_fraction)))
    gap = max(2, int(round(n * gap_fraction)))
    val_start = n - n_val
    train_stop = val_start - gap
    if train_stop < 4:
        raise ValueError("not enough fit probes remain before the inner-validation gap")
    return np.arange(train_stop), np.arange(val_start, n)
