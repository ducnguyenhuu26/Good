"""Orthogonalized stage-contrast baseline for weakly supervised dynamics changes.

The estimator accepts only state/action/velocity-increment observations plus
opaque training group IDs. It does not import the simulator or its interventions.
Action response is an unknown nuisance function approximated by odd polynomial
features; velocity mechanisms are explicit grey-box inductive biases.

Use held-out backgrounds to select regularization. Do not use final test sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


MECHANISMS = ("linear_drag", "quadratic_drag", "position_feedback", "action")


def _check(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x), 2):
        raise ValueError("expected x=[N,6], delta_v=[N,2]")
    if len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid or insufficient observed transitions")
    return x, y


def action_features(a: np.ndarray, max_degree: int = 5) -> np.ndarray:
    """Odd monomials in two observed action coordinates; no simulator formula."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("actions must be [N,2]")
    if max_degree not in (1, 3, 5, 7):
        raise ValueError("choose an odd degree 1,3,5,7")
    columns = [a[:, 0] ** i * a[:, 1] ** (degree - i)
               for degree in range(1, max_degree + 1, 2)
               for i in range(degree + 1)]
    return np.stack(columns, axis=1)


def design(x: np.ndarray, max_degree: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved [2N,p] action nuisance and [2N,3] velocity mechanisms."""
    x = np.asarray(x, dtype=np.float64)
    a = action_features(x[:, 4:6], max_degree)
    n, p = a.shape
    action = np.zeros((2 * n, 2 * p), dtype=np.float64)
    action[0::2, :p] = a
    action[1::2, p:] = a
    q, v = x[:, :2], x[:, 2:4]
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    mech = np.column_stack(((-v).reshape(-1), (-speed * v).reshape(-1), (-q).reshape(-1)))
    return action, mech


def _ridge_solve(mat: np.ndarray, target: np.ndarray, penalty: np.ndarray,
                 center: np.ndarray) -> tuple[np.ndarray, float, float]:
    normal = mat.T @ mat + penalty
    rhs = mat.T @ target + penalty @ center
    coefficients = np.linalg.solve(normal + np.eye(normal.shape[0]) * 1e-10, rhs)
    residual = target - mat @ coefficients
    return coefficients, float(residual @ residual), float(np.linalg.cond(normal))


def _regularization(p: int, degree: int, action_ridge: float, physics_ridge: float,
                    physics_prior: np.ndarray | None, n_extra: int = 0,
                    extra_penalty: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """A and B action axes share degree-dependent ridge; physical terms have optional priors."""
    degrees = [d for d in range(1, degree + 1, 2) for _ in range(d + 1)]
    action_penalty = [action_ridge * (d / 1.0) ** 2 for d in degrees] * 2
    vals = action_penalty * (p // (2 * len(degrees))) + [physics_ridge] * 3 + [extra_penalty] * n_extra
    mu = np.zeros(len(vals))
    if physics_prior is not None:
        mu[-(3+n_extra):-(n_extra) if n_extra else None] = np.asarray(physics_prior)
    return np.diag(vals), mu


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    delta: float
    rss: float
    bic: float
    condition: float
    mechanism: str


def fit_old(x, y, *, dt: float = .05, max_degree: int = 5,
            action_ridge: float = .05, physics_ridge: float = 0.0,
            physics_prior=None) -> tuple[np.ndarray, float]:
    x, y = _check(x, y)
    a, p = design(x, max_degree)
    matrix = np.concatenate((a, p), axis=1)
    penalty, center = _regularization(a.shape[1], max_degree, action_ridge,
                                     physics_ridge, physics_prior)
    coef, _, cond = _ridge_solve(matrix, y.reshape(-1) / dt, penalty, center)
    return coef[-3:], cond


def fit_pair(old_x, old_y, after_x, after_y, mechanism: str, *,
             dt: float = .05, sigma: float = .0005, max_degree: int = 5,
             action_ridge: float = .05, physics_ridge: float = 0.,
             physics_prior=None) -> Fit:
    """Profile the same unknown action response across stages for velocity changes.

    For an action-change candidate, stage-specific nuisance features are allowed.
    BIC penalizes the extra nuisance coefficients, preventing free overfitting.
    This procedure uses observed before/after transitions, not paired oracle effects.
    """
    if mechanism not in MECHANISMS:
        raise ValueError("unknown candidate mechanism")
    old_x, old_y = _check(old_x, old_y)
    after_x, after_y = _check(after_x, after_y)
    a0, p0 = design(old_x, max_degree)
    a1, p1 = design(after_x, max_degree)
    na = a0.shape[1]
    if mechanism == "action":
        old_matrix = np.concatenate((a0, np.zeros_like(a0), p0), axis=1)
        new_matrix = np.concatenate((np.zeros_like(a1), a1, p1), axis=1)
        matrix = np.concatenate((old_matrix, new_matrix), axis=0)
        # 2 action functions, each with its own nuisance coefficients.
        base_penalty, base_center = _regularization(2 * na, max_degree, action_ridge,
                              physics_ridge, physics_prior)
        penalty = base_penalty
        center = base_center
        n_params = 2 * na + 3
    else:
        j = MECHANISMS.index(mechanism)
        before = np.concatenate((a0, p0, np.zeros((len(p0), 1))), axis=1)
        delta = p1[:, j:j+1]
        after = np.concatenate((a1, p1, delta), axis=1)
        matrix = np.concatenate((before, after), axis=0)
        penalty, center = _regularization(na, max_degree, action_ridge,
                                         physics_ridge, physics_prior, n_extra=1)
        n_params = na + 4
    target = np.concatenate((old_y.reshape(-1), after_y.reshape(-1))) / dt
    coef, rss, cond = _ridge_solve(matrix, target, penalty, center)
    # Known simulator measurement noise is part of the public benchmark protocol.
    # BIC compares models on the same observed transitions, with a shared scale.
    nobs = len(target)
    bic = rss / (sigma / dt) ** 2 + n_params * np.log(nobs)
    phys_index = -3 if mechanism == "action" else -4
    beta = coef[phys_index:phys_index+3]
    return Fit(beta, float(coef[-1]) if mechanism != "action" else 0.,
               rss, float(bic), cond, mechanism)


@dataclass(frozen=True)
class LearnedCode:
    mechanism: str
    ratio: float
    mean_score_by_mechanism: Mapping[str, float]
    valid_backgrounds: int


def learn_opaque_codes(grouped_pairs: Mapping[int, list[tuple]], *,
                       dt: float = .05, sigma: float = .0005,
                       max_degree: int = 5, action_ridge: float = .05,
                       ratio_bounds: tuple[float, float] = (.5, 2.0)) -> dict[int, LearnedCode]:
    """Learn one mechanism assignment and magnitude per opaque code, without code semantics.

    Each pair is (old_x, old_delta_v, new_x, new_delta_v), on a distinct
    training background. Do not pass simulator theta, code meanings or queries.
    """
    learned = {}
    for code, backgrounds in grouped_pairs.items():
        if not backgrounds:
            raise ValueError("every opaque intervention requires training pairs")
        fitted = {m: [fit_pair(*pair, m, dt=dt, sigma=sigma,
                        max_degree=max_degree, action_ridge=action_ridge)
                      for pair in backgrounds] for m in MECHANISMS}
        scores = {m: float(np.mean([f.bic for f in rows])) for m, rows in fitted.items()}
        best = min(MECHANISMS, key=lambda name: scores[name])
        ratios = []
        if best != "action":
            j = MECHANISMS.index(best)
            for f in fitted[best]:
                if f.beta[j] > 1e-4 and np.isfinite(f.condition):
                    ratio = 1.0 + f.delta / f.beta[j]
                    if np.isfinite(ratio):
                        ratios.append(np.clip(ratio, *ratio_bounds))
        learned[int(code)] = LearnedCode(best, float(np.median(ratios)) if ratios else 1.,
                                         scores, len(ratios))
    return learned


def fit_joint_velocity(old_x, old_y, after_x, after_y, *, mechanism: str,
                       ratio: float, dt: float = .05, max_degree: int = 5,
                       action_ridge: float = .05, physics_ridge: float = 0.,
                       physics_prior=None, sigma: float = .0005,
                       return_standard_error: bool = False):
    """Use old support plus N new-stage transitions to infer pre-stage coefficients.

    No test program identity or privileged physics parameters are consumed;
    callers may run this for every *candidate* code and compare held-out scores.
    """
    if mechanism not in MECHANISMS[:3] or ratio <= 0:
        raise ValueError("joint fit requires a positive velocity/position scaling")
    old_x, old_y = _check(old_x, old_y)
    after_x, after_y = _check(after_x, after_y)
    a0, p0 = design(old_x, max_degree)
    a1, p1 = design(after_x, max_degree)
    idx = MECHANISMS.index(mechanism)
    pnew = p1.copy()
    pnew[:, idx] *= ratio
    matrix = np.concatenate((np.concatenate((a0,p0), axis=1),
                             np.concatenate((a1,pnew), axis=1)), axis=0)
    penalty, center = _regularization(a0.shape[1], max_degree, action_ridge,
                                     physics_ridge, physics_prior)
    target = np.concatenate((old_y.reshape(-1), after_y.reshape(-1))) / dt
    coef, _, cond = _ridge_solve(matrix, target, penalty, center)
    if return_standard_error:
        normal = matrix.T @ matrix + penalty + 1e-10 * np.eye(matrix.shape[1])
        standard_errors = (sigma / dt) * np.sqrt(np.maximum(
            np.diag(np.linalg.inv(normal))[-3:], 0.0))
        return coef[-3:], cond, standard_errors
    return coef[-3:], cond


def robust_empirical_bayes(estimate: float, se: float, prior_mean: float,
                           prior_std: float, *, slab_scale: float = .12,
                           slab_weight: float = .1) -> tuple[float, float]:
    """Gaussian spike-and-slab predictive shrinkage for distribution shift.

    The broad component is a mechanism-generic robustness prior, not an OOD
    class label. All inputs must be estimated from observed training/support
    data. Returns posterior mean and broad-component responsibility.
    """
    from math import exp, log, pi
    if se <= 0 or prior_std <= 0 or slab_scale <= 0 or not 0 < slab_weight < 1:
        raise ValueError('positive uncertainties and mixture weight in (0,1) required')
    narrow_var = prior_std ** 2
    broad_var = slab_scale ** 2
    means = []
    scores = []
    for tau2, weight in ((narrow_var, 1 - slab_weight), (broad_var, slab_weight)):
        marginal_variance = tau2 + se ** 2
        score = log(weight) - .5 * (log(2*pi*marginal_variance) +
                                    (estimate - prior_mean) ** 2 / marginal_variance)
        mean = (prior_mean / tau2 + estimate / (se * se)) / (1/tau2 + 1/(se*se))
        means.append(mean)
        scores.append(score)
    likelihood_ratio = max(-60.0, min(60.0, scores[0] - scores[1]))
    broad_posterior = 1.0 / (1.0 + exp(likelihood_ratio))
    return ((1.0 - broad_posterior) * means[0] + broad_posterior * means[1],
            broad_posterior)


def predict_velocity_effect(probes: np.ndarray, *, mechanism: str, before_coefficient: float,
                            ratio: float, dt: float = .05) -> np.ndarray:
    """Execute a learned velocity mechanism code on independent [N,6] probes."""
    if mechanism not in MECHANISMS[:3]:
        raise ValueError("velocity effect requires one of three mechanisms")
    x = np.asarray(probes, dtype=np.float64)
    _, p = design(x)
    j = MECHANISMS.index(mechanism)
    return (dt * p[:, j].reshape(-1, 2) * float(before_coefficient) * (float(ratio)-1.))
