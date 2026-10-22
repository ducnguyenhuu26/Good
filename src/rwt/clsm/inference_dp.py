"""Exact finite-world code/age recursion with a factored strength posterior.

The categorical `(code, age)` state is exact. Conditional Gaussian strength
updates use event-conditioned cubature and moment matching, which is an
explicit assumed-density approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch
from torch import Tensor

from .evolution import LawEvolution


@dataclass
class ExactDPResult:
    log_evidence: Tensor
    filtered: Tensor  # [time, code=8, age]
    smoothed: Tensor  # [time, code=8, age]
    event_counts: dict[str, Tensor]
    pairwise_events: Tensor  # [time-1, code, age, target_code, event]


def run_length_event_forward_backward(log_emissions: Tensor, initial: Tensor,
                                      hazard: Tensor, transition: Tensor,
                                      reset_probability: Tensor) -> ExactDPResult:
    """Exact categorical HSMM smoother with event-conditioned emissions.

    Event axis order is `stay_drift`, `stay_reset`, `switch`. For `t=0`,
    channel zero is used for the initial physical transition. Code 000 reset
    is forced to zero regardless of the supplied vector.
    """
    emissions = torch.as_tensor(log_emissions, dtype=torch.float64)
    initial = torch.as_tensor(initial, dtype=torch.float64, device=emissions.device)
    hazard = torch.as_tensor(hazard, dtype=torch.float64, device=emissions.device)
    transition = torch.as_tensor(transition, dtype=torch.float64, device=emissions.device)
    rho = torch.as_tensor(reset_probability, dtype=torch.float64, device=emissions.device).clone()
    if emissions.ndim != 3 or emissions.shape[1:] != (8, 3) or emissions.shape[0] < 1:
        raise ValueError("event emissions must have shape [T,8,3]")
    horizon = emissions.shape[0]
    if initial.ndim != 2 or initial.shape[0] != 8 or not initial.shape[1]:
        raise ValueError("initial age distribution must have shape [8,U]")
    if rho.shape != (8,) or torch.any((rho < 0) | (rho >= 1)):
        raise ValueError("reset probabilities must have shape [8] and lie in [0,1)")
    rho[0] = 0.
    if transition.shape != (8, 8) or not torch.allclose(transition.sum(-1), torch.ones(8, dtype=torch.float64, device=emissions.device), atol=1e-12, rtol=0) or torch.any(torch.diagonal(transition) != 0):
        raise ValueError("structural transition must be row stochastic without self transitions")
    if torch.any((hazard < 0) | (hazard > 1)) or torch.any(initial < 0) or not torch.isclose(initial.sum(), torch.tensor(1., dtype=torch.float64, device=initial.device), atol=1e-12, rtol=0):
        raise ValueError("invalid initial distribution or hazard")
    support = torch.nonzero(initial.sum(0) > 0, as_tuple=True)[0]
    if not len(support):
        raise ValueError("initial age distribution has empty support")
    max_initial_age = int(support[-1])
    width = max_initial_age + horizon
    if hazard.shape[0] != 8 or hazard.shape[1] < max_initial_age + horizon - 1:
        raise ValueError("hazard does not cover reachable age support")
    log_trans = torch.where(transition > 0, transition.log(), -torch.inf)
    log_h = torch.where(hazard > 0, hazard.log(), -torch.inf)
    log_stay = torch.log1p(-hazard)
    log_drift = torch.log1p(-rho)
    log_reset = torch.where(rho > 0, rho.log(), -torch.inf)
    log_a = torch.full((horizon, 8, width), -torch.inf, dtype=torch.float64, device=emissions.device)
    initial_n = min(initial.shape[1], width)
    log_a[0, :, :initial_n] = torch.where(initial[:, :initial_n] > 0,
                                           initial[:, :initial_n].log(), -torch.inf)
    log_a[0] += emissions[0, :, 0, None]
    scales = [torch.logsumexp(log_a[0].flatten(), 0)]
    log_a[0] -= scales[0]
    for t in range(1, horizon):
        active_width = max_initial_age + t
        prev = log_a[t - 1, :, :active_width]
        nxt = torch.full_like(log_a[0], -torch.inf)
        drift = prev + log_stay[:, :active_width] + log_drift[:, None] + emissions[t, :, 0, None]
        reset = prev + log_stay[:, :active_width] + log_reset[:, None] + emissions[t, :, 1, None]
        nxt[:, 1:active_width + 1] = torch.logaddexp(drift, reset)
        switch_parent = torch.logsumexp(prev + log_h[:, :active_width], dim=1)
        switch_target = torch.logsumexp(
            switch_parent[:, None] + log_trans + emissions[t, :, 2][None, :], dim=0)
        nxt[:, 0] = switch_target
        scale = torch.logsumexp(nxt.flatten(), 0)
        log_a[t] = nxt - scale
        scales.append(scale)
    log_b = torch.full_like(log_a, -torch.inf)
    log_b[-1] = 0.
    for t in range(horizon - 2, -1, -1):
        active_width = max_initial_age + t + 1
        next_target = emissions[t + 1, :, 2] + log_b[t + 1, :, 0]
        switch = torch.logsumexp(log_trans + next_target[None, :], dim=1)
        drift = (log_stay[:, :active_width] + log_drift[:, None] +
                 emissions[t + 1, :, 0, None] + log_b[t + 1, :, 1:active_width + 1])
        reset = (log_stay[:, :active_width] + log_reset[:, None] +
                 emissions[t + 1, :, 1, None] + log_b[t + 1, :, 1:active_width + 1])
        stay = torch.logaddexp(drift, reset)
        switch_age = log_h[:, :active_width] + switch[:, None]
        log_b[t, :, :active_width] = torch.logaddexp(stay, switch_age) - scales[t + 1]
    smoothed_log = log_a + log_b
    smoothed_log -= torch.logsumexp(smoothed_log.flatten(1), dim=1)[:, None, None]
    pairwise = torch.zeros((horizon - 1, 8, width, 8, 3), dtype=torch.float64, device=emissions.device)
    eye = torch.eye(8, dtype=torch.float64, device=emissions.device)
    for t in range(horizon - 1):
        active_width = max_initial_age + t + 1
        parent = log_a[t, :, :active_width]
        common_stay = (parent + log_stay[:, :active_width] + log_drift[:, None] +
                       emissions[t + 1, :, 0, None] + log_b[t + 1, :, 1:active_width + 1] - scales[t + 1])
        pairwise[t, :, :active_width, :, 0] = common_stay.exp()[:, :, None] * eye[:, None, :]
        reset_log_mass = (parent + log_stay[:, :active_width] + log_reset[:, None] +
                          emissions[t + 1, :, 1, None] + log_b[t + 1, :, 1:active_width + 1] - scales[t + 1])
        pairwise[t, :, :active_width, :, 1] = reset_log_mass.exp()[:, :, None] * eye[:, None, :]
        switch_mass = (parent[:, :, None] + log_h[:, :active_width, None] + log_trans[:, None, :] +
                       (emissions[t + 1, :, 2] + log_b[t + 1, :, 0])[None, None, :] - scales[t + 1])
        pairwise[t, :, :active_width, :, 2] = switch_mass.exp()
    event_counts = {name: pairwise[..., idx].sum()
                    for idx, name in enumerate(("stay_drift", "stay_reset", "switch"))}
    return ExactDPResult(torch.stack(scales).sum(), log_a.exp(), smoothed_log.exp(),
                         event_counts, pairwise)


def run_length_forward_backward(log_emissions: Tensor, initial: Tensor,
                                hazard: Tensor, transition: Tensor) -> ExactDPResult:
    """Exact categorical HSMM forward/backward for emissions indexed by code.

    `initial` is [8, U], `hazard` is [8, U], and `transition` is [8, 8]
    with zero diagonal and each off-diagonal row summing to one.
    """
    emissions = torch.as_tensor(log_emissions, dtype=torch.float64)
    initial = torch.as_tensor(initial, dtype=torch.float64, device=emissions.device)
    hazard = torch.as_tensor(hazard, dtype=torch.float64, device=emissions.device)
    transition = torch.as_tensor(transition, dtype=torch.float64, device=emissions.device)
    if emissions.ndim != 2 or emissions.shape[1] != 8 or emissions.shape[0] < 1:
        raise ValueError("log_emissions must have shape [T,8], T>=1")
    horizon = emissions.shape[0]
    if initial.ndim != 2 or initial.shape[0] != 8 or not initial.shape[1]:
        raise ValueError("initial age distribution must have shape [8,U]")
    initial_support = torch.nonzero(initial.sum(0) > 0, as_tuple=True)[0]
    if not len(initial_support):
        raise ValueError("initial age distribution has empty support")
    width = int(initial_support[-1]) + horizon
    if hazard.ndim != 2 or hazard.shape[0] != 8 or hazard.shape[1] < width - 1:
        raise ValueError("hazard must provide every reachable code-age value")
    if transition.shape != (8, 8) or not torch.allclose(
            transition.sum(-1), torch.ones(8, dtype=torch.float64, device=transition.device),
            atol=1e-12, rtol=0):
        raise ValueError("structural transition matrix must be row stochastic")
    if torch.any(torch.diagonal(transition) != 0):
        raise ValueError("structural transition must exclude self transitions")
    if torch.any(initial < 0) or not torch.isclose(initial.sum(), torch.tensor(1., dtype=torch.float64, device=initial.device), atol=1e-12, rtol=0):
        raise ValueError("initial age distribution must be a probability measure")
    if torch.any((hazard < 0) | (hazard > 1)):
        raise ValueError("hazards must lie in [0,1]")

    log_alpha = torch.full((horizon, 8, width), -torch.inf, dtype=torch.float64, device=emissions.device)
    initial_width = min(width, initial.shape[1])
    log_alpha[0, :, :initial_width] = torch.where(
        initial[:, :initial_width] > 0, initial[:, :initial_width].log(), -torch.inf)
    log_alpha[0] += emissions[0, :, None]
    scales = torch.logsumexp(log_alpha[0].flatten(), 0)
    log_alpha[0] -= scales
    log_scales = [scales]
    log_transition = torch.where(transition > 0, transition.log(), -torch.inf)
    log_hazard = torch.where(hazard > 0, hazard.log(), -torch.inf)
    log_stay = torch.log1p(-hazard)
    for t in range(1, horizon):
        prev = log_alpha[t - 1]
        nxt = torch.full_like(prev, -torch.inf)
        max_age = min(width, int(initial_support[-1]) + t)
        stay_mass = prev[:, :max_age] + log_stay[:, :max_age]
        nxt[:, 1:max_age + 1] = stay_mass
        switch_by_parent = torch.logsumexp(prev[:, :max_age] + log_hazard[:, :max_age], dim=1)
        switch_by_target = torch.logsumexp(switch_by_parent[:, None] + log_transition, dim=0)
        nxt[:, 0] = switch_by_target
        nxt += emissions[t, :, None]
        scale = torch.logsumexp(nxt.flatten(), 0)
        log_alpha[t] = nxt - scale
        log_scales.append(scale)

    # Backward messages use the same normalized forward convention.
    log_beta = torch.full_like(log_alpha, -torch.inf)
    log_beta[-1] = 0.
    for t in range(horizon - 2, -1, -1):
        max_age = min(width, int(initial_support[-1]) + t + 1)
        next_emission_beta = emissions[t + 1] + log_beta[t + 1, :, 0]
        switch_value = torch.logsumexp(log_transition + next_emission_beta[None, :], dim=1)
        stay = log_stay[:, :max_age] + emissions[t + 1, :, None] + log_beta[t + 1, :, 1:max_age + 1]
        switch = log_hazard[:, :max_age] + switch_value[:, None]
        log_beta[t, :, :max_age] = torch.logaddexp(stay, switch) - log_scales[t + 1]
    log_gamma = log_alpha + log_beta
    log_gamma -= torch.logsumexp(log_gamma.flatten(1), dim=1)[:, None, None]
    smoothed = log_gamma.exp()

    pairwise = torch.zeros((max(horizon - 1, 0), 8, width, 8, 2), dtype=torch.float64, device=emissions.device)
    for t in range(horizon - 1):
        max_age = min(width, int(initial_support[-1]) + t + 1)
        parent = log_alpha[t, :, :max_age]
        stay_log = (parent + log_stay[:, :max_age] + emissions[t + 1, :, None] +
                    log_beta[t + 1, :, 1:max_age + 1] - log_scales[t + 1])
        pairwise[t, :, :max_age, :, 0] = (
            stay_log.exp()[:, :, None] * torch.eye(8, dtype=torch.float64, device=emissions.device)[:, None, :])
        switch_to_target = emissions[t + 1] + log_beta[t + 1, :, 0]
        switch_log = (parent[:, :, None] + log_hazard[:, :max_age, None] +
                      log_transition[:, None, :] + switch_to_target[None, None, :] - log_scales[t + 1])
        pairwise[t, :, :max_age, :, 1] = switch_log.exp()
    event_counts = {"stay": pairwise[..., 0].sum(), "switch": pairwise[..., 1].sum()}
    log_evidence = torch.stack(log_scales).sum()
    return ExactDPResult(log_evidence, log_alpha.exp(), smoothed, event_counts, pairwise)


@dataclass
class _EventBranch:
    code: int
    age: int
    event: str
    log_weight: Tensor
    mean: Tensor
    covariance: Tensor
    parent_code: int
    parent_age: int
    parent_mean: Tensor | None = None
    parent_covariance: Tensor | None = None


def _normal_log_prob(residual: Tensor, covariance: Tensor, jitter: float = 1e-10) -> Tensor:
    covariance = (covariance + covariance.T) * 0.5
    eye = torch.eye(len(residual), dtype=residual.dtype, device=residual.device)
    last = None
    for attempt in range(6):
        chol, info = torch.linalg.cholesky_ex(covariance + jitter * (10 ** attempt) * eye)
        if int(info) == 0:
            solve = torch.cholesky_solve(residual[:, None], chol).squeeze(-1)
            return -0.5 * (residual @ solve + 2 * chol.diagonal().log().sum() +
                           len(residual) * math.log(2 * math.pi))
        last = int(info)
    raise RuntimeError(f"predictive covariance Cholesky failed; info={last}")


def cubature_correct(model, obs: Tensor, action: Tensor, next_obs: Tensor,
                     code: int, mean: Tensor, covariance: Tensor,
                     jitter: float = 1e-10) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Positive-weight 2p cubature Kalman correction on active coordinates."""
    device, dtype = obs.device, obs.dtype
    mean = mean.to(device=device, dtype=dtype)
    covariance = covariance.to(device=device, dtype=dtype)
    active = torch.tensor([(code >> j) & 1 for j in range(3)], dtype=torch.bool, device=device)
    indices = active.nonzero(as_tuple=True)[0]
    p = len(indices)
    sigma = [mean.clone()]
    weights: Tensor
    if p:
        sub = covariance[indices[:, None], indices[None, :]]
        sub = (sub + sub.T) * 0.5
        eye = torch.eye(p, dtype=dtype, device=device)
        chol = None
        for attempt in range(6):
            candidate, info = torch.linalg.cholesky_ex(sub + jitter * (10 ** attempt) * eye)
            if int(info) == 0:
                chol = candidate
                break
        if chol is None:
            raise RuntimeError("cubature prior covariance Cholesky failed")
        sigma = []
        for j in range(p):
            delta = math.sqrt(p) * chol[:, j]
            plus, minus = mean.clone(), mean.clone()
            plus[indices] += delta
            minus[indices] -= delta
            sigma.extend((plus, minus))
        weights = torch.full((2 * p,), 1.0 / (2 * p), dtype=dtype, device=device)
    else:
        weights = torch.ones(1, dtype=dtype, device=device)
    theta = torch.stack(sigma)
    observations = obs.expand(len(sigma), -1)
    actions = action.expand(len(sigma), -1)
    codes = torch.full((len(sigma),), code, dtype=torch.long, device=device)
    predicted_mean, predicted_var = model(observations, actions, codes, theta)
    ybar = (weights[:, None] * predicted_mean).sum(0)
    centered_y = predicted_mean - ybar
    predictive_cov = torch.diag((weights[:, None] * predicted_var).sum(0))
    predictive_cov += torch.einsum("n,ni,nj->ij", weights, centered_y, centered_y)
    residual = next_obs - ybar
    log_evidence = _normal_log_prob(residual, predictive_cov, jitter)
    updated_mean, updated_cov = mean.clone(), covariance.clone()
    if p:
        centered_x = theta[:, indices] - mean[indices]
        cross = torch.einsum("n,ni,nj->ij", weights, centered_x, centered_y)
        gain = torch.linalg.solve(predictive_cov, cross.T).T
        updated_mean[indices] = mean[indices] + gain @ residual
        post = covariance[indices[:, None], indices[None, :]] - gain @ predictive_cov @ gain.T
        post = (post + post.T) * 0.5
        eigval, eigvec = torch.linalg.eigh(post)
        post = (eigvec * eigval.clamp_min(0)) @ eigvec.T
        updated_cov[indices[:, None], indices[None, :]] = post
        inactive = (~active).nonzero(as_tuple=True)[0]
        updated_cov[indices[:, None], inactive[None, :]] = 0
        updated_cov[inactive[:, None], indices[None, :]] = 0
        updated_mean[inactive] = 0
        updated_cov[inactive[:, None], inactive[None, :]] = 0
    return log_evidence, updated_mean, updated_cov, {
        "sigma_points": len(sigma), "jitter": float(jitter),
        "predictive_mean": ybar.detach(), "predictive_covariance": predictive_cov.detach(),
    }


class AgeLawFilter:
    """Exact discrete `(code, age)` filter with per-code Gaussian strengths.

    The categorical age recursion is untruncated. Strength is represented by
    one Gaussian per code; correlations with age are approximated by explicit
    event-conditioned cubature and code-level moment aggregation.
    """

    def __init__(self, model, evolution: LawEvolution, memory, max_world_length: int = 500,
                 initial_law_information: str = "known_nominal"):
        if max_world_length < 1:
            raise ValueError("max_world_length must be positive")
        if initial_law_information not in ("known_nominal", "unknown", "comp_design"):
            raise ValueError("initial_law_information must be known_nominal, unknown, or comp_design")
        self.model, self.evolution, self.memory = model, evolution, memory
        self.max_world_length = int(max_world_length)
        self.time = 0
        self.log_code_age = torch.full((8, max_world_length + 1), -torch.inf, dtype=torch.float64)
        if initial_law_information == "known_nominal":
            self.log_code_age[0, 0] = 0.
        elif initial_law_information == "unknown":
            self.log_code_age[:, 0] = -math.log(8.)
        else:
            # The registered COMP calibration design draws nominal and each
            # singleton mechanism with equal probability; this is public prior
            # information, not a realized evaluator label.
            self.log_code_age[[0, 1, 2, 4], 0] = -math.log(4.)
        self.xi_mean = torch.zeros((8, 3), dtype=torch.float64)
        self.xi_cov = torch.zeros((8, 3, 3), dtype=torch.float64)
        for code in range(8):
            mean, cov = memory.entry_prior(code)
            self.xi_mean[code] = mean.detach().to(torch.float64)
            self.xi_cov[code] = cov.detach().to(torch.float64)
        self.pending: list[_EventBranch] | None = None
        self.last_events: list[dict] = []
        self.diagnostics: list[dict] = []

    @property
    def probabilities(self) -> Tensor:
        return torch.where(torch.isfinite(self.log_code_age), self.log_code_age.exp(), 0.)

    def observe(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> dict:
        if self.pending is None:
            branches = [_EventBranch(code, age, "initial", self.log_code_age[code, age],
                                     self.xi_mean[code], self.xi_cov[code], code, age,
                                     self.xi_mean[code], self.xi_cov[code])
                        for code in range(8) for age in range(self.time + 1)
                        if torch.isfinite(self.log_code_age[code, age])]
        else:
            branches = self.pending
        if not branches:
            raise RuntimeError("age filter has no supported branches")
        # Each age branch of a shared (code,event) template receives identical
        # physical evidence, while preserving its separate categorical mass.
        cache: dict[tuple[int, str, bytes, bytes], tuple[Tensor, Tensor, Tensor, dict]] = {}
        corrected = []
        for branch in branches:
            key = (branch.code, branch.event,
                   branch.mean.detach().cpu().numpy().tobytes(),
                   branch.covariance.detach().cpu().numpy().tobytes())
            if key not in cache:
                cache[key] = cubature_correct(self.model, obs, action, next_obs,
                                              branch.code, branch.mean, branch.covariance)
            evidence, mean, covariance, info = cache[key]
            corrected.append((branch, branch.log_weight + evidence, mean, covariance, info))
        normalizer = torch.logsumexp(torch.stack([row[1] for row in corrected]), 0)
        new_log = torch.full_like(self.log_code_age, -torch.inf)
        by_code: dict[int, list[tuple[Tensor, Tensor, Tensor]]] = {code: [] for code in range(8)}
        events = []
        max_jitter = 0.
        completed: dict[int, list[tuple[Tensor, Tensor, Tensor, Tensor]]] = {}
        for branch, logw, mean, covariance, info in corrected:
            normalized_logw = logw - normalizer
            new_log[branch.code, branch.age] = torch.logaddexp(
                new_log[branch.code, branch.age], normalized_logw)
            weight = normalized_logw.exp()
            by_code[branch.code].append((weight, mean, covariance))
            max_jitter = max(max_jitter, info["jitter"])
            events.append({"code": branch.code, "age": branch.age, "event": branch.event,
                           "probability": float(weight.detach()), "parent_code": branch.parent_code,
                           "parent_age": branch.parent_age})
            if branch.event in ("stay_reset", "switch") and branch.parent_code != 0 and branch.parent_mean is not None:
                completed.setdefault(branch.parent_code, []).append(
                    (weight, branch.parent_mean, branch.parent_covariance, mean))
        for parent_code, rows in completed.items():
            responsibility = torch.stack([row[0] for row in rows]).sum()
            conditional = [row[0] / responsibility for row in rows]
            parent_mean = sum((weight * row[1] for row, weight in zip(rows, conditional)),
                              torch.zeros(3, dtype=torch.float64, device=obs.device))
            parent_cov = torch.zeros((3, 3), dtype=torch.float64, device=obs.device)
            for row, weight in zip(rows, conditional):
                delta = row[1] - parent_mean
                parent_cov += weight * (row[2] + torch.outer(delta, delta))
            if hasattr(self.memory, "record_completed_visit"):
                self.memory.record_completed_visit(self.time, parent_code, parent_mean,
                                                   parent_cov, float(responsibility))
        for code, rows in by_code.items():
            if not rows:
                continue
            mass = torch.stack([row[0] for row in rows]).sum()
            weights = [row[0] / mass for row in rows]
            mean = sum((w * row[1] for row, w in zip(rows, weights)), torch.zeros(3, dtype=torch.float64, device=obs.device))
            covariance = torch.zeros((3, 3), dtype=torch.float64, device=obs.device)
            for row, weight in zip(rows, weights):
                delta = row[1] - mean
                covariance += weight * (row[2] + torch.outer(delta, delta))
            self.xi_mean[code] = mean
            self.xi_cov[code] = (covariance + covariance.T) * 0.5
        self.log_code_age = new_log
        self.pending = None
        self.last_events = events
        output = {"time": self.time, "predictive_log_likelihood": float(normalizer.detach()),
                  "code_posterior": self.probabilities.sum(1).tolist(),
                  "age_posterior": self.probabilities.sum(0).tolist(),
                  "event_posterior": events, "max_cubature_jitter": max_jitter,
                  "mass": float(self.probabilities.sum().detach())}
        self.diagnostics.append(output)
        return output

    def evolve(self) -> dict:
        if self.pending is not None:
            raise RuntimeError("observe must precede evolve")
        if self.time >= self.max_world_length:
            raise RuntimeError("maximum finite world length exceeded")
        probs = self.probabilities
        rho = torch.stack([self.evolution.reset_probability(code).detach().to(self.log_code_age.device)
                           for code in range(8)])
        q = self.evolution.q.detach().to(torch.float64)
        transition = self.evolution.structural_log_transition().detach().exp()
        branches = []
        next_log = torch.full_like(self.log_code_age, -torch.inf)
        for code in range(8):
            active = torch.tensor([(code >> j) & 1 for j in range(3)], dtype=torch.float64)
            for age in range(self.time + 1):
                mass = probs[code, age]
                if float(mass.detach()) <= 0:
                    continue
                hazard = self.evolution.hazard(torch.tensor([code]), torch.tensor([age]))[0].detach()
                log_mass = self.log_code_age[code, age]
                # stay with continuous drift
                drift_prob = (1 - hazard) * (1 - rho[code])
                if float(drift_prob) > 0:
                    mean = self.xi_mean[code].clone()
                    cov = self.xi_cov[code] + torch.diag(q * active)
                    branches.append(_EventBranch(code, age + 1, "stay_drift",
                                                 log_mass + drift_prob.log(), mean, cov, code, age,
                                                 self.xi_mean[code].clone(), self.xi_cov[code].clone()))
                # Same-code reset closes a parameter visit but preserves age.
                reset_prob = (1 - hazard) * rho[code]
                if float(reset_prob) > 0:
                    mean, cov = self.memory.entry_prior(code)
                    branches.append(_EventBranch(code, age + 1, "stay_reset",
                                                 log_mass + reset_prob.log(), mean, cov, code, age,
                                                 self.xi_mean[code].clone(), self.xi_cov[code].clone()))
                # Structural switches begin age zero and use target entry prior.
                if float(hazard) > 0:
                    for target in range(8):
                        probability = hazard * transition[code, target]
                        if target == code or float(probability) <= 0:
                            continue
                        mean, cov = self.memory.entry_prior(target)
                        branches.append(_EventBranch(target, 0, "switch",
                                                     log_mass + probability.log(), mean, cov,
                                                     code, age, self.xi_mean[code].clone(),
                                                     self.xi_cov[code].clone()))
        for branch in branches:
            next_log[branch.code, branch.age] = torch.logaddexp(
                next_log[branch.code, branch.age], branch.log_weight)
        mass = torch.logsumexp(next_log.flatten(), 0)
        if not torch.isfinite(mass) or abs(float(mass.detach())) > 1e-10:
            raise RuntimeError(f"exact code-age mass conservation failed: log_mass={float(mass.detach()):.3g}")
        self.log_code_age = next_log - mass
        self.pending = [_EventBranch(b.code, b.age, b.event, b.log_weight - mass,
                                     b.mean, b.covariance, b.parent_code, b.parent_age,
                                     b.parent_mean, b.parent_covariance)
                        for b in branches]
        self.time += 1
        return {"time": self.time, "mass": float(self.probabilities.sum().detach()),
                "structural_event_mass": sum(float(b.log_weight.exp().detach()) for b in branches if b.event == "switch"),
                "parameter_reset_event_mass": sum(float(b.log_weight.exp().detach()) for b in branches if b.event == "stay_reset"),
                "branch_count": len(branches)}

    def clone_causal(self) -> "AgeLawFilter":
        clone = AgeLawFilter(self.model, self.evolution, copy.deepcopy(self.memory), self.max_world_length,
                             "unknown")
        clone.time = self.time
        clone.log_code_age = self.log_code_age.clone()
        clone.xi_mean, clone.xi_cov = self.xi_mean.clone(), self.xi_cov.clone()
        clone.pending = None if self.pending is None else [
            _EventBranch(b.code, b.age, b.event, b.log_weight.clone(), b.mean.clone(),
                         b.covariance.clone(), b.parent_code, b.parent_age,
                         None if b.parent_mean is None else b.parent_mean.clone(),
                         None if b.parent_covariance is None else b.parent_covariance.clone()) for b in self.pending]
        return clone
