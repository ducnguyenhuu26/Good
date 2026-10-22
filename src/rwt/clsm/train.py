"""Nominal initialization and public-data approximate CLSM E-step.

The M-step and full checkpoint selection are still release gated. An E-step
alone does not constitute trained CLSM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .belief import LawComponent, LawFilter, StepAncestry
from .diagnostics import evolution_diagnostic, inference_diagnostic
from .memory import TheoryMemory
from .smoothing import GaussianSmoothingResult, backward_component_weights, backward_gaussian_moments


@dataclass
class EStepResult:
    log_predictive: Tensor
    filtered: list[list[LawComponent]]
    ancestry: list[StepAncestry]
    smoothed: GaussianSmoothingResult
    gamma: list[Tensor]
    max_dropped_mass: float
    memory: TheoryMemory
    drift_variance: Tensor
    inference_diagnostics: list[dict[str, float]]
    evolution_diagnostics: list[dict[str, float]]


def approximate_estep(filter_model: LawFilter, obs: Tensor, action: Tensor,
                      next_obs: Tensor) -> EStepResult:
    """Run emission-then-evolution filtering and an event-aware backward pass.

    The three arrays describe one independent public world sequence. Callers
    must construct a fresh filter and memory for each world boundary.
    """
    if obs.ndim != 2 or action.ndim != 2 or next_obs.shape != obs.shape or len(obs) != len(action) or len(obs) == 0:
        raise ValueError("public world sequence shapes mismatch")
    filtered: list[list[LawComponent]] = []
    ancestry: list[StepAncestry] = []
    evidence = []
    inference_rows: list[dict[str, float]] = []
    evolution_rows: list[dict[str, float]] = []
    for t in range(len(obs)):
        result = filter_model.observe(obs[t], action[t], next_obs[t])
        evidence.append(result["log_predictive"])
        prior_step_loss = ancestry[-1].dropped_mass if ancestry else 0.0
        inference_rows.append(inference_diagnostic(filter_model.components, t, prior_step_loss))
        evolution_rows.append(evolution_diagnostic(filter_model.components, filter_model.evolution, t))
        filtered.append([LawComponent(c.code, c.age, c.mean.clone(), c.cov.clone(),
                                      c.log_weight.clone(), c.scratch_or_reuse)
                         for c in filter_model.components])
        if t + 1 < len(obs):
            ancestry.append(filter_model.evolve())
    weights = [torch.stack([c.log_weight.exp() for c in step]) for step in filtered]
    discrete = backward_component_weights(weights, ancestry)
    gaussian = backward_gaussian_moments(filtered, ancestry, discrete,
                                         filter_model.evolution.q.detach(),
                                         filter_model.jitter_initial, filter_model.jitter_max)
    return EStepResult(torch.tensor(evidence, dtype=torch.float64), filtered, ancestry,
                       gaussian, discrete.gamma,
                       max((step.dropped_mass for step in ancestry), default=0.0),
                       filter_model.memory, filter_model.evolution.q.detach().clone(),
                       inference_rows, evolution_rows)


@dataclass
class PublicSequence:
    obs: Tensor
    action: Tensor
    next_obs: Tensor


@dataclass
class EmissionStatistics:
    obs: Tensor
    action: Tensor
    next_obs: Tensor
    code: Tensor
    theta: Tensor
    cov: Tensor
    weight: Tensor


def collect_emission_statistics(sequences: list[PublicSequence],
                                posteriors: list[EStepResult]) -> EmissionStatistics:
    """Stop gradients at approximate E-step posterior sufficient statistics."""
    if len(sequences) != len(posteriors) or not sequences:
        raise ValueError("one posterior required per public sequence")
    rows = []
    for sequence, posterior in zip(sequences, posteriors):
        if len(sequence.obs) != len(posterior.gamma):
            raise ValueError("posterior/sequence length mismatch")
        for t, components in enumerate(posterior.filtered):
            for k, component in enumerate(components):
                weight = posterior.gamma[t][k]
                if float(weight) <= 0:
                    continue
                rows.append((sequence.obs[t], sequence.action[t], sequence.next_obs[t],
                             component.code, posterior.smoothed.mean[t][k],
                             posterior.smoothed.cov[t][k], weight))
    if not rows:
        raise RuntimeError("E-step has no positive posterior support")
    return EmissionStatistics(torch.stack([r[0] for r in rows]).detach(),
                              torch.stack([r[1] for r in rows]).detach(),
                              torch.stack([r[2] for r in rows]).detach(),
                              torch.tensor([r[3] for r in rows], dtype=torch.long),
                              torch.stack([r[4] for r in rows]).detach(),
                              torch.stack([r[5] for r in rows]).detach(),
                              torch.stack([r[6] for r in rows]).detach())


def expected_emission_nll(model, stats: EmissionStatistics, indices: Tensor | None = None,
                          *, sampled_by_weight: bool = False) -> Tensor:
    """Six-point cubature with an unbiased estimator for either row sampler.

    The full-data objective weights each row by its frozen E-step mass. Rows
    sampled proportionally to that mass use an ordinary mean. Uniformly
    sampled rows use the corresponding fixed-total importance correction.
    """
    is_sample = indices is not None
    if indices is None:
        indices = torch.arange(len(stats.weight))
    codes = stats.code[indices]
    bits = torch.stack([((codes >> j) & 1) for j in range(3)], dim=1).to(torch.float64)
    theta = stats.theta[indices] * bits
    cov = stats.cov[indices] * bits[:, :, None] * bits[:, None, :]
    # A tiny inactive-coordinate jitter is masked again before evaluation.
    eye = torch.eye(3, dtype=torch.float64, device=theta.device)
    chol = torch.linalg.cholesky(cov + 1e-9 * eye)
    delta = math.sqrt(3) * chol.transpose(1, 2)
    points = torch.cat((theta[:, None, :] + delta,
                        theta[:, None, :] - delta), dim=1) * bits[:, None, :]
    count = len(indices)
    logp = model.log_prob(stats.obs[indices].repeat_interleave(6, dim=0),
                          stats.action[indices].repeat_interleave(6, dim=0),
                          stats.next_obs[indices].repeat_interleave(6, dim=0),
                          codes.repeat_interleave(6), points.reshape(count * 6, 3))
    per_component = -logp.reshape(count, 6).mean(1)
    weights = stats.weight[indices]
    total_weight = stats.weight.sum().clamp_min(1e-12)
    if sampled_by_weight:
        if not is_sample:
            raise ValueError("weighted sampling requires explicit row indices")
        return per_component.mean()
    if is_sample:
        return (weights * per_component).mean() * len(stats.weight) / total_weight
    return (weights * per_component).sum() / total_weight


def expected_evolution_nll(evolution, posteriors: list[EStepResult]) -> Tensor:
    """Expected event, destination-code, and active-coordinate drift log loss.

    The drift second moment uses the linear-Gaussian backward conditional for
    each retained stay-drift edge. A merged child Gaussian remains an explicit
    approximation; the E-step statistics are held fixed during this M-step.
    """
    total = torch.zeros((), dtype=torch.float64)
    count = torch.zeros((), dtype=torch.float64)
    log_a = evolution.structural_log_transition()
    safe_log_a = torch.where(torch.isfinite(log_a), log_a, torch.zeros_like(log_a))
    rho = evolution.reset_prob
    q = evolution.q
    for posterior in posteriors:
        for t, step in enumerate(posterior.ancestry):
            parents = posterior.filtered[t]
            children = posterior.filtered[t + 1]
            event_xi = posterior.smoothed.event_xi[t]
            parent_code = torch.tensor([c.code for c in parents], dtype=torch.long)
            parent_age = torch.tensor([c.age for c in parents], dtype=torch.long)
            child_code = torch.tensor([c.code for c in children], dtype=torch.long)
            hazard = evolution.hazard(parent_code, parent_age).clamp(1e-12, 1 - 1e-12)
            drift_mass = event_xi["stay_drift"].sum(1).detach()
            reset_mass = event_xi["stay_reset"].sum(1).detach()
            switch_mass = event_xi["switch"].detach()
            total = total - (drift_mass * (torch.log1p(-hazard) + torch.log1p(-rho))).sum()
            total = total - (reset_mass * (torch.log1p(-hazard) + torch.log(rho))).sum()
            total = total - (switch_mass * (hazard.log()[:, None] +
                         safe_log_a[parent_code[:, None], child_code[None, :]])).sum()
            count = count + drift_mass.sum() + reset_mass.sum() + switch_mass.sum()
            for k, parent in enumerate(parents):
                active = torch.tensor([(parent.code >> j) & 1 for j in range(3)], dtype=torch.bool)
                if not bool(active.any()):
                    continue
                ids = active.nonzero(as_tuple=True)[0]
                prior_cov = parent.cov[ids[:, None], ids[None, :]].detach()
                prior_mean = parent.mean[ids].detach()
                for j in torch.nonzero(event_xi["stay_drift"][k] > 0, as_tuple=True)[0].tolist():
                    mass = event_xi["stay_drift"][k, j].detach()
                    child_mean = posterior.smoothed.mean[t + 1][j][ids].detach()
                    child_cov = posterior.smoothed.cov[t + 1][j][ids[:, None], ids[None, :]].detach()
                    # Conditional moments are fixed E-step quantities.
                    q_old = posterior.drift_variance[ids]
                    predicted_cov = prior_cov + torch.diag(q_old)
                    gain = torch.linalg.solve(predicted_cov, prior_cov).T
                    conditional_cov = prior_cov - gain @ prior_cov
                    identity = torch.eye(len(ids), dtype=torch.float64)
                    difference = (identity - gain) @ (child_mean - prior_mean)
                    difference_cov = (identity - gain) @ child_cov @ (identity - gain).T + conditional_cov
                    second = difference.square() + difference_cov.diag().clamp_min(0)
                    total = total + .5 * mass * (q[ids].log() + second / q[ids]).sum()
    if float(count) <= 0:
        raise ValueError("no evolution transitions in E-step")
    return total / count


def generalized_em_mstep(model, evolution, hierarchy, stats: EmissionStatistics,
                         posteriors: list[EStepResult], config: dict,
                         *, seed: int | None = None, optimizer=None,
                         generator: torch.Generator | None = None) -> list[dict[str, float]]:
    """Bounded stochastic M-step on frozen E-step statistics.

    This does not yet fit hierarchical hyperparameters and is therefore an
    incomplete CLSM trainer. The caller must retain this release boundary.
    """
    cfg = config["training"]
    if generator is None:
        if seed is None:
            raise ValueError("seed or independent training generator required")
        generator = torch.Generator().manual_seed(seed)
    if optimizer is None:
        optimizer = make_mstep_optimizer(model, evolution, hierarchy, config)
    metrics = []
    probabilities = (stats.weight / stats.weight.sum()).to("cpu")
    for _ in range(cfg["gradient_steps_per_iteration"]):
        indices = torch.multinomial(probabilities, cfg["batch_size"],
                                    replacement=True, generator=generator)
        emission = expected_emission_nll(model, stats, indices, sampled_by_weight=True)
        transition = expected_evolution_nll(evolution, posteriors)
        hierarchy_nll = hierarchy.expected_complete_nll([p.memory for p in posteriors])
        penalty = model.mechanics.regularizer(
            config["mechanisms"]["sparse_weight"], config["mechanisms"]["overlap_weight"])
        inflation = (model.log_variance - model.reference_log_variance).clamp_min(0).square().mean()
        loss = emission + transition + hierarchy_nll + penalty + cfg["variance_penalty_weight"] * inflation
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite generalized-EM M-step objective")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(evolution.parameters()) +
                                              list(hierarchy.parameters()),
                                              cfg["gradient_clip_norm"])
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite generalized-EM gradient")
        optimizer.step()
        metrics.append({"emission_nll": float(emission.detach()),
                        "evolution_nll": float(transition.detach()),
                        "hierarchy_nll": float(hierarchy_nll.detach()),
                        "regularizer": float(penalty.detach()),
                        "variance_inflation_penalty": float(inflation.detach()),
                        "gradient_norm": float(norm)})
    return metrics


def make_mstep_optimizer(model, evolution, hierarchy, config: dict):
    cfg = config["training"]
    return torch.optim.Adam([
        {"params": model.parameters(), "lr": cfg["lr_dynamics"]},
        {"params": evolution.parameters(), "lr": cfg["lr_evolution"]},
        {"params": hierarchy.parameters(), "lr": cfg["lr_evolution"]},
    ], weight_decay=cfg["weight_decay"])


def train_reference(model, obs: Tensor, action: Tensor, next_obs: Tensor,
                    *, epochs: int = 10, lr: float = 3e-4, seed: int = 41,
                    batch_size: int = 256, weight_decay: float = 1e-4,
                    gradient_clip_norm: float = 5.0) -> list[float]:
    if epochs < 1 or lr <= 0:
        raise ValueError("positive training budget required")
    if obs.ndim != 2 or action.ndim != 2 or next_obs.shape != obs.shape or len(obs) != len(action):
        raise ValueError("reference transition batch mismatch")
    obs, action, next_obs = (x.to(torch.float64) for x in (obs, action, next_obs))
    # The caller must supply a separately collected, known-nominal reference
    # dataset. No evaluator intervention label enters the model objective.
    model.fit_normalization(obs, action, next_obs)
    generator = torch.Generator(device=obs.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    code = torch.zeros(len(obs), dtype=torch.long, device=obs.device)
    theta = torch.zeros((len(obs),3), dtype=torch.float64, device=obs.device)
    curve = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(obs), generator=generator, device=obs.device)
        for indices in order.split(min(batch_size, len(obs))):
            predicted = model(obs[indices], action[indices], code[indices], theta[indices])[0]
            loss = ((predicted - next_obs[indices]) / model.delta_scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reference loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite reference gradient")
            optimizer.step()
        with torch.no_grad():
            predicted = model(obs, action, code, theta)[0]
            curve.append(float(((predicted - next_obs) / model.delta_scale).square().mean()))
    with torch.no_grad():
        predicted = model(obs, action, code, theta)[0]
        residual_variance = (predicted - next_obs).square().mean(0).clamp(
            min=model.variance_floor, max=model.variance_ceiling)
        model.reference_log_variance.copy_(residual_variance.log())
        model.log_variance.copy_(residual_variance.log())
    return curve
