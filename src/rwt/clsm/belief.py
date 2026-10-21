"""Float64 Rao-Blackwellized mixture filtering with explicit time order."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .evolution import LawEvolution
from .memory import TheoryMemory


@dataclass
class LawComponent:
    code: int
    age: int
    mean: Tensor
    cov: Tensor
    log_weight: Tensor
    scratch_or_reuse: int = 1


def _normal_log_prob(residual: Tensor, chol: Tensor) -> Tensor:
    solved = torch.cholesky_solve(residual[:, None], chol).squeeze(-1)
    return -0.5 * (residual @ solved + 2 * chol.diag().log().sum() + len(residual) * math.log(2 * math.pi))


def _chol_with_jitter(matrix: Tensor) -> tuple[Tensor, float]:
    eye = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    for _ in range(8):
        try:
            return torch.linalg.cholesky(matrix + jitter * eye), jitter
        except RuntimeError:
            jitter = 1e-9 if jitter == 0 else 10 * jitter
    raise RuntimeError(f"Cholesky failed at jitter {jitter:g}")


def ekf_correct(model, component: LawComponent, obs: Tensor, action: Tensor,
                next_obs: Tensor) -> tuple[LawComponent, Tensor, float]:
    """One nonlinear Gaussian emission update, with Joseph covariance form."""
    mask = torch.tensor([(component.code >> j) & 1 for j in range(3)], dtype=torch.bool,
                        device=component.mean.device)
    m = component.mean.detach().clone()
    c = component.cov.detach().clone()
    active = mask.nonzero(as_tuple=True)[0]
    code = torch.tensor([component.code], dtype=torch.long, device=obs.device)

    def prediction(active_theta: Tensor) -> Tensor:
        theta = torch.zeros(3, dtype=obs.dtype, device=obs.device).scatter(0, active, active_theta)
        return model(obs[None], action[None], code, theta[None])[0][0]

    with torch.enable_grad():
        active_mean = m[active].detach().requires_grad_(True)
        pred = prediction(active_mean)
        jac = torch.autograd.functional.jacobian(prediction, active_mean).detach()
    with torch.no_grad():
        variance = model(obs[None], action[None], code, m[None])[1][0]
        noise = torch.diag(variance)
        residual = next_obs - pred.detach()
        subcov = c[active[:, None], active[None, :]]
        innovation = noise + jac @ subcov @ jac.T
        chol, jitter = _chol_with_jitter(innovation)
        log_likelihood = _normal_log_prob(residual, chol)
        if len(active):
            gain = torch.cholesky_solve(jac @ subcov, chol).T
            updated_mean = m.clone()
            updated_mean[active] = m[active] + gain @ residual
            identity = torch.eye(len(active), dtype=c.dtype, device=c.device)
            left = identity - gain @ jac
            updated_cov = c.clone()
            block = left @ subcov @ left.T + gain @ noise @ gain.T
            updated_cov[active[:, None], active[None, :]] = (block + block.T) / 2
        else:
            updated_mean, updated_cov = m, c
    return LawComponent(component.code, component.age, updated_mean, updated_cov,
                        component.log_weight, component.scratch_or_reuse), log_likelihood, jitter


def compress_components(components: list[LawComponent], budget: int) -> tuple[list[LawComponent], float]:
    """Moment-merge same support only, then preserve each supported code/model.

    Returns discarded probability *before* renormalization. Never allocates a
    slot to zero-mass hypotheses.
    """
    if budget < 1 or not components:
        raise ValueError("positive budget and nonempty components required")
    weights = torch.stack([c.log_weight for c in components]).to(torch.float64)
    weights = (weights - torch.logsumexp(weights, 0)).exp()
    groups: dict[tuple[int, int, int], list[tuple[LawComponent, Tensor]]] = {}
    for item, weight in zip(components, weights):
        if float(weight) <= 0:
            continue
        groups.setdefault((item.code, item.age, item.scratch_or_reuse), []).append((item, weight))
    merged = []
    for (code, age, hypothesis), items in groups.items():
        mass = torch.stack([w for _, w in items]).sum()
        conditional = [w / mass for _, w in items]
        mean = sum((w * item.mean for (item, _), w in zip(items, conditional)), torch.zeros_like(items[0][0].mean))
        cov = torch.zeros_like(items[0][0].cov)
        for (item, _), w in zip(items, conditional):
            diff = item.mean - mean
            cov += w * (item.cov + torch.outer(diff, diff))
        merged.append(LawComponent(code, age, mean, (cov + cov.T) / 2, mass.log(), hypothesis))
    merged.sort(key=lambda item: float(item.log_weight), reverse=True)
    required: dict[tuple[int, int], LawComponent] = {}
    for item in merged:
        required.setdefault((item.code, item.scratch_or_reuse), item)
    if len(required) > budget:
        raise ValueError("budget cannot preserve all positive code/model support")
    retained = list(required.values())
    retained_ids = {id(x) for x in retained}
    retained.extend(x for x in merged if id(x) not in retained_ids and len(retained) < budget)
    kept = torch.stack([x.log_weight.exp() for x in retained]).sum()
    dropped = float((1 - kept).clamp_min(0))
    normalizer = torch.logsumexp(torch.stack([x.log_weight for x in retained]), 0)
    for item in retained:
        item.log_weight -= normalizer
    return retained, dropped


class LawFilter:
    def __init__(self, model, evolution: LawEvolution, memory: TheoryMemory | None = None,
                 budget: int = 128):
        self.model = model
        self.evolution = evolution
        self.memory = TheoryMemory() if memory is None else memory
        self.budget = int(budget)
        if self.budget < 16:
            raise ValueError("budget must preserve eight codes and two prior models")
        self.components: list[LawComponent] = []
        for code in range(8):
            for hypothesis in (0, 1):
                mean, cov = self.memory.prior_for_entry(code, hypothesis)
                self.components.append(LawComponent(code, 0, mean, cov,
                    torch.tensor(-math.log(8) + float(self.memory.log_model_weights[hypothesis]), dtype=torch.float64), hypothesis))
        self.time = 0
        self.needs_evolution = False
        self.diagnostics: list[dict[str, float]] = []

    def predict_current(self) -> list[LawComponent]:
        if self.needs_evolution:
            raise RuntimeError("call evolve before forecasting the next time step")
        return self.components

    def observe(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> dict[str, float]:
        if self.needs_evolution:
            raise RuntimeError("the previous observation needs evolution")
        obs, action, next_obs = (x.to(torch.float64) for x in (obs, action, next_obs))
        if obs.shape != (self.model.schema.observation_dim,) or action.shape != (self.model.schema.action_dim,) or next_obs.shape != obs.shape:
            raise ValueError("public transition shape mismatch")
        updated = []
        logits = []
        jitters = []
        for component in self.components:
            corrected, emission, jitter = ekf_correct(self.model, component, obs, action, next_obs)
            updated.append(corrected)
            logits.append(component.log_weight + emission)
            jitters.append(jitter)
        logits = torch.stack(logits)
        evidence = torch.logsumexp(logits, 0)
        for component, value in zip(updated, logits - evidence):
            component.log_weight = value.detach()
        self.components = updated
        # Predictive evidence is computed from the pre-observation prior.
        result = {"time": float(self.time), "log_predictive": float(evidence),
                  "max_cholesky_jitter": max(jitters)}
        self.diagnostics.append(result)
        self.needs_evolution = True
        return result

    def evolve(self) -> None:
        if not self.needs_evolution:
            raise RuntimeError("observe the physical transition before evolution")
        branches = []
        for component in self.components:
            for branch in self.evolution.predict_branches(component, self.memory):
                if branch.probability > 0:
                    branches.append(LawComponent(branch.code, branch.age, branch.mean, branch.cov,
                        component.log_weight + math.log(branch.probability), component.scratch_or_reuse))
        self.components, dropped = compress_components(branches, self.budget)
        self.diagnostics[-1]["dropped_mass"] = dropped
        self.diagnostics[-1]["inference_degraded"] = float(dropped > 1e-3)
        self.time += 1
        self.needs_evolution = False

    def state_dict(self) -> dict:
        return {"time": self.time, "needs_evolution": self.needs_evolution,
                "components": [{"code": c.code, "age": c.age, "mean": c.mean.clone(),
                                "cov": c.cov.clone(), "log_weight": c.log_weight.clone(),
                                "scratch_or_reuse": c.scratch_or_reuse} for c in self.components],
                "memory": {"primitive_mean": self.memory.primitive_mean.clone(),
                           "primitive_var": self.memory.primitive_var.clone(),
                           "theory_mean": {k: v.clone() for k, v in self.memory.theory_mean.items()},
                           "theory_var": {k: v.clone() for k, v in self.memory.theory_var.items()},
                           "log_model_weights": self.memory.log_model_weights.clone(),
                           "completed_visits": self.memory.completed_visits}}
