"""Nominal initialization and public-data approximate CLSM E-step.

The M-step and full checkpoint selection are still release gated. An E-step
alone does not constitute trained CLSM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .belief import LawComponent, LawFilter, StepAncestry
from .smoothing import GaussianSmoothingResult, backward_component_weights, backward_gaussian_moments


@dataclass
class EStepResult:
    log_predictive: Tensor
    filtered: list[list[LawComponent]]
    ancestry: list[StepAncestry]
    smoothed: GaussianSmoothingResult
    gamma: list[Tensor]
    max_dropped_mass: float


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
    for t in range(len(obs)):
        result = filter_model.observe(obs[t], action[t], next_obs[t])
        evidence.append(result["log_predictive"])
        filtered.append([LawComponent(c.code, c.age, c.mean.clone(), c.cov.clone(),
                                      c.log_weight.clone(), c.scratch_or_reuse)
                         for c in filter_model.components])
        if t + 1 < len(obs):
            ancestry.append(filter_model.evolve())
    weights = [torch.stack([c.log_weight.exp() for c in step]) for step in filtered]
    discrete = backward_component_weights(weights, ancestry)
    gaussian = backward_gaussian_moments(filtered, ancestry, discrete,
                                         filter_model.evolution.q.detach())
    return EStepResult(torch.tensor(evidence, dtype=torch.float64), filtered, ancestry,
                       gaussian, discrete.gamma,
                       max((step.dropped_mass for step in ancestry), default=0.0))


def train_reference(model, obs: Tensor, action: Tensor, next_obs: Tensor,
                    *, epochs: int = 10, lr: float = 3e-4, seed: int = 41) -> list[float]:
    if epochs < 1 or lr <= 0:
        raise ValueError("positive training budget required")
    if obs.ndim != 2 or action.ndim != 2 or next_obs.shape != obs.shape or len(obs) != len(action):
        raise ValueError("reference transition batch mismatch")
    obs, action, next_obs = (x.to(torch.float64) for x in (obs, action, next_obs))
    # The caller must supply a separately collected, known-nominal reference
    # dataset. No evaluator intervention label enters the model objective.
    model.fit_normalization(obs, action, next_obs)
    generator = torch.Generator(device=obs.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    code = torch.zeros(len(obs), dtype=torch.long, device=obs.device)
    theta = torch.zeros((len(obs),3), dtype=torch.float64, device=obs.device)
    curve = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(obs), generator=generator, device=obs.device)
        for indices in order.split(min(256, len(obs))):
            predicted = model(obs[indices], action[indices], code[indices], theta[indices])[0]
            loss = ((predicted - next_obs[indices]) / model.delta_scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reference loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite reference gradient")
            optimizer.step()
        with torch.no_grad():
            predicted = model(obs, action, code, theta)[0]
            curve.append(float(((predicted - next_obs) / model.delta_scale).square().mean()))
    return curve
