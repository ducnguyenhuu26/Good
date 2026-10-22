"""Float64 Rao-Blackwellized mixture filtering with explicit time order."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .evolution import LawEvolution
from .memory import TheoryMemory, VisitSummary


@dataclass
class LawComponent:
    code: int
    age: int
    mean: Tensor
    cov: Tensor
    log_weight: Tensor
    scratch_or_reuse: int = 1


@dataclass
class StepAncestry:
    """Retained finite-mixture transition weights for an offline backward pass."""

    filtered_weight: Tensor  # [K] after observation t
    predicted_weight: Tensor  # [J] before observation t+1
    transition: Tensor       # [K,J], proposal/pruning corrected
    dropped_mass: float
    event_transition: dict[str, Tensor] | None = None


def _normal_log_prob(residual: Tensor, chol: Tensor) -> Tensor:
    solved = torch.cholesky_solve(residual[:, None], chol).squeeze(-1)
    return -0.5 * (residual @ solved + 2 * chol.diag().log().sum() + len(residual) * math.log(2 * math.pi))


def _chol_with_jitter(matrix: Tensor, initial: float = 1e-9,
                      maximum: float = 1e-3) -> tuple[Tensor, float]:
    if not 0 < initial <= maximum:
        raise ValueError("invalid Cholesky jitter range")
    eye = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    while jitter <= maximum:
        try:
            return torch.linalg.cholesky(matrix + jitter * eye), jitter
        except RuntimeError:
            jitter = initial if jitter == 0 else jitter * 10
    raise RuntimeError(f"Cholesky failed at jitter {jitter:g}")


def ekf_correct(model, component: LawComponent, obs: Tensor, action: Tensor,
                next_obs: Tensor, jitter_initial: float = 1e-9,
                jitter_max: float = 1e-3) -> tuple[LawComponent, Tensor, float]:
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
        chol, jitter = _chol_with_jitter(innovation, jitter_initial, jitter_max)
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
                 budget: int = 128, max_dropped_mass: float = 1e-3,
                 jitter_initial: float = 1e-9, jitter_max: float = 1e-3):
        self.model = model
        self.evolution = evolution
        self.memory = TheoryMemory() if memory is None else memory
        self.budget = int(budget)
        self.max_dropped_mass = float(max_dropped_mass)
        self.jitter_initial = float(jitter_initial)
        self.jitter_max = float(jitter_max)
        if not 0 < self.jitter_initial <= self.jitter_max:
            raise ValueError("invalid Cholesky jitter range")
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
        prior_by_model = []
        for hypothesis in (0, 1):
            values = [c.log_weight for c in self.components if c.scratch_or_reuse == hypothesis]
            prior_by_model.append(torch.logsumexp(torch.stack(values), 0) if values else
                                  torch.tensor(-torch.inf, dtype=torch.float64))
        self.memory.log_model_weights = torch.stack(prior_by_model).detach()
        for component in self.components:
            corrected, emission, jitter = ekf_correct(self.model, component, obs, action, next_obs,
                                                       self.jitter_initial, self.jitter_max)
            updated.append(corrected)
            logits.append(component.log_weight + emission)
            jitters.append(jitter)
        logits = torch.stack(logits)
        evidence = torch.logsumexp(logits, 0)
        predictive_by_model = []
        for hypothesis in (0, 1):
            selected = [value for c, value in zip(self.components, logits)
                        if c.scratch_or_reuse == hypothesis]
            predictive_by_model.append(float(torch.logsumexp(torch.stack(selected), 0) - prior_by_model[hypothesis])
                                       if selected and bool(torch.isfinite(prior_by_model[hypothesis])) else -torch.inf)
        for component, value in zip(updated, logits - evidence):
            component.log_weight = value.detach()
        self.components = updated
        self.memory.update_prequential_evidence(*predictive_by_model)
        posterior_by_model = torch.stack([
            torch.logsumexp(torch.stack([c.log_weight for c in self.components if c.scratch_or_reuse == hypothesis]), 0)
            if any(c.scratch_or_reuse == hypothesis for c in self.components) else
            torch.tensor(-torch.inf, dtype=torch.float64)
            for hypothesis in (0, 1)])
        if not torch.allclose(self.memory.log_model_weights, posterior_by_model, atol=1e-10):
            raise RuntimeError("scratch/reuse evidence was counted inconsistently")
        # Predictive evidence is computed from the pre-observation prior.
        result = {"time": float(self.time), "log_predictive": float(evidence),
                  "max_cholesky_jitter": max(jitters)}
        self.diagnostics.append(result)
        self.needs_evolution = True
        return result

    def _update_completed_visit_memory(self) -> float:
        """One soft visit-level summary per code at its posterior switch mass.

        This is an online moment approximation to the hierarchical posterior.
        Within-visit transitions are filtered into one latent-parameter
        posterior; only expected *structural exits* enter prototype precision.
        Responsibilities are conditional on the scratch/reuse hypothesis, not
        on the code. The code probability must remain in each precision update.
        Same-code parameter resets do not generate visit updates.
        """
        total_mass = 0.
        total_weight = torch.stack([c.log_weight.exp() for c in self.components]).sum()
        for hypothesis in (0, 1):
            hypothesis_weights = [c.log_weight.exp() for c in self.components
                                  if c.scratch_or_reuse == hypothesis]
            if not hypothesis_weights:
                continue
            hypothesis_mass = torch.stack(hypothesis_weights).sum()
            if float(hypothesis_mass) <= 0:
                continue
            model_weight = float((hypothesis_mass / total_weight).clamp(0, 1))
            for code in range(1, 8):
                members = [c for c in self.components
                           if c.code == code and c.scratch_or_reuse == hypothesis]
                if not members:
                    continue
                masses = torch.stack([c.log_weight.exp() / hypothesis_mass * self.evolution.hazard(
                    torch.tensor([c.code]), torch.tensor([c.age]))[0].detach() for c in members])
                mass = masses.sum()
                if float(mass) <= 0:
                    continue
                normalized = masses / mass
                mean = sum((w * c.mean for w, c in zip(normalized, members)),
                           torch.zeros(3, dtype=torch.float64))
                cov = torch.zeros((3, 3), dtype=torch.float64)
                for w, c in zip(normalized, members):
                    diff = c.mean - mean
                    cov += w * (c.cov + torch.outer(diff, diff))
                self.memory.update_from_completed_visit(
                    VisitSummary(code, mean, cov, float(mass), hypothesis, model_weight))
                total_mass += float(mass) * model_weight
        return total_mass

    def evolve(self) -> StepAncestry:
        if not self.needs_evolution:
            raise RuntimeError("observe the physical transition before evolution")
        visit_mass = self._update_completed_visit_memory()
        branches = []
        origins: list[tuple[int, float, str]] = []
        corrected_weights = torch.stack([c.log_weight.exp() for c in self.components])
        for parent, component in enumerate(self.components):
            for branch in self.evolution.predict_branches(component, self.memory):
                if branch.probability > 0:
                    branches.append(LawComponent(branch.code, branch.age, branch.mean, branch.cov,
                        component.log_weight + math.log(branch.probability), component.scratch_or_reuse))
                    origins.append((parent, branch.probability, branch.event))
        self.components, dropped = compress_components(branches, self.budget)
        key_to_column = {(c.code, c.age, c.scratch_or_reuse): j for j, c in enumerate(self.components)}
        transition = torch.zeros((len(corrected_weights), len(self.components)), dtype=torch.float64)
        event_transition = {event: torch.zeros_like(transition) for event in ("stay_drift", "stay_reset", "switch")}
        for branch, (parent, probability, event) in zip(branches, origins):
            column = key_to_column.get((branch.code, branch.age, branch.scratch_or_reuse))
            if column is not None:
                transition[parent, column] += probability / (1 - dropped)
                event_transition[event][parent, column] += probability / (1 - dropped)
        predicted = torch.stack([c.log_weight.exp() for c in self.components])
        if not torch.allclose(corrected_weights @ transition, predicted, atol=1e-8):
            raise RuntimeError("ancestry weights do not reproduce predicted mixture")
        if not torch.allclose(sum(event_transition.values()), transition, atol=1e-12):
            raise RuntimeError("event ancestry does not sum to total ancestry")
        self.diagnostics[-1]["dropped_mass"] = dropped
        self.diagnostics[-1]["completed_visit_mass"] = visit_mass
        self.diagnostics[-1]["inference_degraded"] = float(dropped > self.max_dropped_mass)
        self.time += 1
        self.needs_evolution = False
        return StepAncestry(corrected_weights, predicted, transition, dropped, event_transition)

    def state_dict(self) -> dict:
        return {"time": self.time, "needs_evolution": self.needs_evolution,
                "components": [{"code": c.code, "age": c.age, "mean": c.mean.clone(),
                                "cov": c.cov.clone(), "log_weight": c.log_weight.clone(),
                                "scratch_or_reuse": c.scratch_or_reuse} for c in self.components],
                "memory": {"prior_mean": self.memory.prior_mean,
                           "prior_var": self.memory.prior_var,
                           "interaction_var": self.memory.interaction_var,
                           "recurrence_var": self.memory.recurrence_var,
                           "primitive_mean": self.memory.primitive_mean.clone(),
                           "primitive_var": self.memory.primitive_var.clone(),
                           "scratch_primitive_mean": self.memory.scratch_primitive_mean.clone(),
                           "scratch_primitive_var": self.memory.scratch_primitive_var.clone(),
                           "theory_mean": {k: v.clone() for k, v in self.memory.theory_mean.items()},
                           "theory_var": {k: v.clone() for k, v in self.memory.theory_var.items()},
                           "scratch_theory_mean": {k: v.clone() for k, v in self.memory.scratch_theory_mean.items()},
                           "scratch_theory_var": {k: v.clone() for k, v in self.memory.scratch_theory_var.items()},
                           "log_model_weights": self.memory.log_model_weights.clone(),
                           "completed_visits": self.memory.completed_visits,
                           "completed_summaries": [{"code": s.code, "mean": s.mean.clone(),
                               "cov": s.cov.clone(), "responsibility": s.responsibility,
                               "scratch_or_reuse": s.scratch_or_reuse,
                               "model_weight": s.model_weight}
                               for s in self.memory.completed_summaries]}}

    def load_state_dict(self, state: dict) -> None:
        """Restore a causal posterior snapshot without consulting the evaluator."""
        if not isinstance(state, dict) or not isinstance(state.get("components"), list):
            raise ValueError("invalid filter state")
        time = state.get("time")
        needs_evolution = state.get("needs_evolution")
        if not isinstance(time, int) or time < 0 or not isinstance(needs_evolution, bool):
            raise ValueError("invalid filter time state")
        components = []
        for item in state["components"]:
            code, age, hypothesis = item["code"], item["age"], item["scratch_or_reuse"]
            mean, cov, log_weight = item["mean"], item["cov"], item["log_weight"]
            if (not isinstance(code, int) or not 0 <= code <= 7 or
                    not isinstance(age, int) or age < 0 or hypothesis not in (0, 1) or
                    mean.shape != (3,) or cov.shape != (3, 3) or log_weight.ndim != 0):
                raise ValueError("invalid filter component state")
            components.append(LawComponent(code, age, mean.detach().clone().to(torch.float64),
                cov.detach().clone().to(torch.float64), log_weight.detach().clone().to(torch.float64), hypothesis))
        if not components or len(components) > self.budget:
            raise ValueError("filter state component count violates configured budget")
        weights = torch.stack([c.log_weight for c in components])
        if not torch.isfinite(torch.logsumexp(weights, 0)) or not torch.allclose(
                torch.logsumexp(weights, 0), torch.zeros((), dtype=torch.float64), atol=1e-8):
            raise ValueError("filter state weights are not normalized")
        memory_state = state.get("memory")
        if not isinstance(memory_state, dict):
            raise ValueError("filter state is missing memory posterior")
        for name in ("prior_mean", "prior_var", "interaction_var", "recurrence_var"):
            setattr(self.memory, name, float(memory_state[name]))
        for name in ("primitive_mean", "primitive_var", "scratch_primitive_mean", "scratch_primitive_var",
                     "log_model_weights"):
            value = memory_state[name]
            expected = (2,) if name == "log_model_weights" else (3,)
            if value.shape != expected:
                raise ValueError(f"invalid memory tensor shape for {name}")
            setattr(self.memory, name, value.detach().clone().to(torch.float64))
        for name in ("theory_mean", "theory_var", "scratch_theory_mean", "scratch_theory_var"):
            values = memory_state[name]
            if any(not 0 <= int(key) <= 7 or value.shape != (3,) for key, value in values.items()):
                raise ValueError(f"invalid memory table for {name}")
            setattr(self.memory, name, {int(key): value.detach().clone().to(torch.float64)
                                        for key, value in values.items()})
        self.memory.completed_visits = float(memory_state["completed_visits"])
        self.memory.completed_summaries = [VisitSummary(
            int(row["code"]), row["mean"].detach().clone().to(torch.float64),
            row["cov"].detach().clone().to(torch.float64), float(row["responsibility"]),
            int(row["scratch_or_reuse"]), float(row["model_weight"]))
            for row in memory_state["completed_summaries"]]
        self.components = components
        self.time = time
        self.needs_evolution = needs_evolution
        self.diagnostics = []
