"""Causal Gaussian primitive/theory memory and prequential scratch comparator."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisitSummary:
    code: int
    mean: Tensor
    cov: Tensor
    responsibility: float = 1.0
    scratch_or_reuse: int = 1
    model_weight: float = 1.0


class TheoryMemory:
    """Approximate posterior; one completed visit contributes one update."""

    def __init__(self, prior_mean: float = -2.0, prior_std: float = 1.5,
                 interaction_var: float = 0.25, recurrence_var: float = 0.25,
                 scratch_weight: float = 0.5):
        if not 0 < scratch_weight <= 1 or min(prior_std, interaction_var, recurrence_var) <= 0:
            raise ValueError("invalid hierarchy prior")
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_std) ** 2
        self.interaction_var = float(interaction_var)
        self.recurrence_var = float(recurrence_var)
        self.primitive_mean = torch.full((3,), prior_mean, dtype=torch.float64)
        self.primitive_var = torch.full((3,), self.prior_var, dtype=torch.float64)
        self.theory_mean: dict[int, Tensor] = {}
        self.theory_var: dict[int, Tensor] = {}
        # Scratch and reuse are predictive models with separate causal
        # posteriors. Scratch begins at the hierarchy prior and adapts online.
        self.scratch_primitive_mean = self.primitive_mean.clone()
        self.scratch_primitive_var = self.primitive_var.clone()
        self.scratch_theory_mean: dict[int, Tensor] = {}
        self.scratch_theory_var: dict[int, Tensor] = {}
        self.log_model_weights = torch.tensor([scratch_weight, 1 - scratch_weight], dtype=torch.float64).log()
        # Expected completed-visit count, not the number of soft update calls.
        self.completed_visits = 0.0
        self.completed_summaries: list[VisitSummary] = []

    @staticmethod
    def _mask(code: int) -> Tensor:
        if not 0 <= code <= 7:
            raise ValueError("invalid code")
        return torch.tensor([(code >> j) & 1 for j in range(3)], dtype=torch.float64)

    def prior_for_entry(self, code: int, scratch_or_reuse: int = 1) -> tuple[Tensor, Tensor]:
        mask = self._mask(code)
        if scratch_or_reuse not in (0, 1):
            raise ValueError("scratch_or_reuse must be 0 or 1")
        if scratch_or_reuse == 0:
            primitive_mean, primitive_var = self.scratch_primitive_mean, self.scratch_primitive_var
            theory_mean, theory_var = self.scratch_theory_mean, self.scratch_theory_var
        else:
            primitive_mean, primitive_var = self.primitive_mean, self.primitive_var
            theory_mean, theory_var = self.theory_mean, self.theory_var
        mean = theory_mean.get(code, primitive_mean)
        var = theory_var.get(code, primitive_var + self.interaction_var)
        return mean * mask, torch.diag((var + self.recurrence_var) * mask)

    def update_from_completed_visit(self, visit_summary: VisitSummary) -> None:
        code = visit_summary.code
        mask = self._mask(code).bool()
        if visit_summary.mean.shape != (3,) or visit_summary.cov.shape != (3, 3):
            raise ValueError("visit posterior shape mismatch")
        if not 0 <= visit_summary.responsibility <= 1:
            raise ValueError("responsibility outside probability range")
        if visit_summary.scratch_or_reuse not in (0, 1):
            raise ValueError("visit summary model must be scratch (0) or reuse (1)")
        if not 0 <= visit_summary.model_weight <= 1:
            raise ValueError("visit summary model weight outside probability range")
        if not bool(mask.any()) or visit_summary.responsibility == 0:
            return
        # One precision increment per visit, never one per transition.
        if visit_summary.scratch_or_reuse == 0:
            primitive_mean, primitive_var = self.scratch_primitive_mean, self.scratch_primitive_var
            theory_mean, theory_var = self.scratch_theory_mean, self.scratch_theory_var
        else:
            primitive_mean, primitive_var = self.primitive_mean, self.primitive_var
            theory_mean, theory_var = self.theory_mean, self.theory_var
        prev_mean = theory_mean.get(code, primitive_mean.clone()).clone()
        prev_var = theory_var.get(code, (primitive_var + self.interaction_var).clone()).clone()
        noise = visit_summary.cov.diag().clamp_min(0) + self.recurrence_var
        precision = prev_var.reciprocal() + visit_summary.responsibility / noise
        updated_var = precision.reciprocal()
        updated_mean = updated_var * (prev_mean / prev_var + visit_summary.responsibility * visit_summary.mean / noise)
        theory_mean[code] = torch.where(mask, updated_mean, torch.zeros_like(updated_mean))
        theory_var[code] = torch.where(mask, updated_var, torch.zeros_like(updated_var))
        # Primitive posterior receives one soft, visit-level observation.
        p_precision = primitive_var.reciprocal() + visit_summary.responsibility / (noise + self.interaction_var)
        p_var = p_precision.reciprocal()
        p_mean = p_var * (primitive_mean / primitive_var + visit_summary.responsibility * visit_summary.mean / (noise + self.interaction_var))
        if visit_summary.scratch_or_reuse == 0:
            self.scratch_primitive_mean = torch.where(mask, p_mean, self.scratch_primitive_mean)
            self.scratch_primitive_var = torch.where(mask, p_var, self.scratch_primitive_var)
        else:
            self.primitive_mean = torch.where(mask, p_mean, self.primitive_mean)
            self.primitive_var = torch.where(mask, p_var, self.primitive_var)
        self.completed_visits += visit_summary.responsibility * visit_summary.model_weight
        self.completed_summaries.append(VisitSummary(code, visit_summary.mean.detach().clone(),
                                                     visit_summary.cov.detach().clone(),
                                                     visit_summary.responsibility,
                                                     visit_summary.scratch_or_reuse,
                                                     visit_summary.model_weight))

    def update_prequential_evidence(self, scratch_log_predictive: float, reuse_log_predictive: float) -> None:
        """Call once *after* the transition whose predictive densities these are."""
        self.log_model_weights += torch.tensor([scratch_log_predictive, reuse_log_predictive], dtype=torch.float64)
        self.log_model_weights -= torch.logsumexp(self.log_model_weights, dim=0)

    def scratch_copy(self) -> "TheoryMemory":
        return TheoryMemory(self.prior_mean, self.prior_var ** 0.5,
                            self.interaction_var, self.recurrence_var, 1.)


class HierarchyParameters(nn.Module):
    """Global Gaussian hierarchy hyperparameters fitted in the GEM M-step."""

    def __init__(self, prior_mean: float = -2.0, prior_std: float = 1.5,
                 interaction_var: float = .25, recurrence_var: float = .25,
                 scratch_weight: float = .5):
        super().__init__()
        if min(prior_std, interaction_var, recurrence_var) <= 0:
            raise ValueError("positive hierarchy scales required")
        self.prior_mean = nn.Parameter(torch.tensor(prior_mean, dtype=torch.float64))
        self.prior_std_raw = nn.Parameter(torch.tensor(math.log(math.expm1(prior_std)), dtype=torch.float64))
        self.interaction_raw = nn.Parameter(torch.tensor(math.log(math.expm1(interaction_var)), dtype=torch.float64))
        self.recurrence_raw = nn.Parameter(torch.tensor(math.log(math.expm1(recurrence_var)), dtype=torch.float64))
        self.scratch_weight = float(scratch_weight)

    @property
    def prior_var(self) -> Tensor:
        return F.softplus(self.prior_std_raw).square() + 1e-6

    @property
    def interaction_var(self) -> Tensor:
        return F.softplus(self.interaction_raw) + 1e-6

    @property
    def recurrence_var(self) -> Tensor:
        return F.softplus(self.recurrence_raw) + 1e-6

    def new_memory(self) -> TheoryMemory:
        return TheoryMemory(float(self.prior_mean.detach()),
                            math.sqrt(float(self.prior_var.detach())),
                            float(self.interaction_var.detach()),
                            float(self.recurrence_var.detach()), self.scratch_weight)

    def expected_complete_nll(self, memories: list[TheoryMemory]) -> Tensor:
        """Approximate Gaussian complete-data hierarchy loss on frozen E-step posteriors.

        The mean-field expectation omits prototype cross covariance; each
        completed visit enters once with its posterior structural-exit mass.
        """
        loss = torch.zeros((), dtype=torch.float64)
        count = torch.zeros((), dtype=torch.float64)
        prior_var = self.prior_var
        interaction = self.interaction_var
        recurrence = self.recurrence_var
        for memory in memories:
            models = (
                (memory.scratch_primitive_mean, memory.scratch_primitive_var,
                 memory.scratch_theory_mean, memory.scratch_theory_var),
                (memory.primitive_mean, memory.primitive_var,
                 memory.theory_mean, memory.theory_var),
            )
            for hypothesis, (primitive_mean_value, primitive_var_value, theory_means, theory_vars) in enumerate(models):
                model_weight = memory.log_model_weights[hypothesis].exp().detach()
                primitive_mean = primitive_mean_value.detach()
                primitive_var = primitive_var_value.detach()
                model_loss = .5 * (3 * prior_var.log() +
                                    ((primitive_mean - self.prior_mean).square() + primitive_var).sum() / prior_var)
                model_count = torch.tensor(3., dtype=torch.float64)
                for code, theory_mean in theory_means.items():
                    active = TheoryMemory._mask(code).bool()
                    theory_var = theory_vars[code].detach()[active]
                    theory_mean = theory_mean.detach()[active]
                    primitive = primitive_mean[active]
                    primitive_uncertainty = primitive_var[active]
                    model_loss = model_loss + .5 * (active.sum() * interaction.log() +
                            ((theory_mean - primitive).square() + theory_var + primitive_uncertainty).sum() / interaction)
                    model_count = model_count + active.sum()
                loss = loss + model_weight * model_loss
                count = count + model_weight * model_count
            for summary in memory.completed_summaries:
                active = TheoryMemory._mask(summary.code).bool()
                if summary.scratch_or_reuse == 0:
                    means, variances = memory.scratch_theory_mean, memory.scratch_theory_var
                    primitive_mean, primitive_var = memory.scratch_primitive_mean, memory.scratch_primitive_var
                else:
                    means, variances = memory.theory_mean, memory.theory_var
                    primitive_mean, primitive_var = memory.primitive_mean, memory.primitive_var
                prototype = means.get(summary.code, primitive_mean).detach()[active]
                prototype_var = variances.get(summary.code, primitive_var).detach()[active]
                visit_mean = summary.mean.detach()[active]
                visit_var = summary.cov.detach().diag()[active]
                responsibility = summary.responsibility * summary.model_weight
                loss = loss + .5 * responsibility * (active.sum() * recurrence.log() +
                         ((visit_mean - prototype).square() + visit_var + prototype_var).sum() / recurrence)
                count = count + responsibility * active.sum()
        return loss / count.clamp_min(1)
