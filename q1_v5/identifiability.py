"""Local efficient-information diagnostics for the frozen V5 response.

The parameterization uses log(b), so the returned information is for log(b).
Shared action-response parameters are treated as fixed; this is conditional
information and therefore an optimistic upper bound when those parameters are
also uncertain.
"""
from __future__ import annotations

import numpy as np


def efficient_information(jac: np.ndarray, target_column: int, *, sigma: float,
                           rtol: float | None = None) -> dict[str, float | int]:
    """Project a target Jacobian column off nuisance columns using SVD.

    `jac` contains stacked output derivatives, with rows as observations and
    columns as parameters. No normal-matrix inverse is formed.
    """
    jac = np.asarray(jac, dtype=np.float64)
    if jac.ndim != 2 or not np.isfinite(jac).all():
        raise ValueError("jac must be a finite 2D array")
    if not 0 <= target_column < jac.shape[1] or sigma <= 0:
        raise ValueError("invalid target column or noise scale")
    z = jac[:, target_column]
    nuisance = np.delete(jac, target_column, axis=1)
    if nuisance.shape[1]:
        u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
        tol = (rtol if rtol is not None else max(nuisance.shape) * np.finfo(float).eps)
        cutoff = tol * (s[0] if len(s) else 0.0)
        rank = int(np.sum(s > cutoff))
        basis = u[:, :rank]
        residual = z - basis @ (basis.T @ z)
        condition = float(s[0] / s[rank - 1]) if rank else float("inf")
    else:
        s = np.asarray([])
        rank = 0
        condition = 1.0
        residual = z.copy()
    residual_norm = float(np.linalg.norm(residual))
    projection_tol = (max(jac.shape) * np.finfo(float).eps *
                      max(float(np.linalg.norm(z)), 1.0))
    information = (0.0 if residual_norm <= projection_tol else
                   float(residual @ residual / sigma**2))
    variance_bound = float(1.0 / information) if information > 0 else float("inf")
    return {
        "efficient_information": information,
        "crlb_variance": variance_bound,
        "crlb_se": float(np.sqrt(variance_bound)),
        "target_norm": float(np.linalg.norm(z)),
        "residual_target_norm": float(np.linalg.norm(residual)),
        "nuisance_rank": rank,
        "nuisance_columns": int(nuisance.shape[1]),
        "nuisance_condition": condition,
        "svd_tolerance": float(cutoff if nuisance.shape[1] else 0.0),
    }
