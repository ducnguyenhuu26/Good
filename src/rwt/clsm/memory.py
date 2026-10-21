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
            mean = torch.full((3,), self.prior_mean, dtype=torch.float64)
            var = torch.full((3,), self.prior_var + self.interaction_var, dtype=torch.float64)
        else:
            mean = self.theory_mean.get(code, self.primitive_mean)
            var = self.theory_var.get(code, self.primitive_var + self.interaction_var)
        return mean * mask, torch.diag((var + self.recurrence_var) * mask)

    def update_from_completed_visit(self, visit_summary: VisitSummary) -> None:
        code = visit_summary.code
        mask = self._mask(code).bool()
        if visit_summary.mean.shape != (3,) or visit_summary.cov.shape != (3, 3):
            raise ValueError("visit posterior shape mismatch")
        if not 0 <= visit_summary.responsibility <= 1:
            raise ValueError("responsibility outside probability range")
        if not bool(mask.any()) or visit_summary.responsibility == 0:
            return
        # One precision increment per visit, never one per transition.
        prev_mean = self.theory_mean.get(code, self.primitive_mean.clone()).clone()
        prev_var = self.theory_var.get(code, (self.primitive_var + self.interaction_var).clone()).clone()
        noise = visit_summary.cov.diag().clamp_min(0) + self.recurrence_var
        precision = prev_var.reciprocal() + visit_summary.responsibility / noise
        updated_var = precision.reciprocal()
        updated_mean = updated_var * (prev_mean / prev_var + visit_summary.responsibility * visit_summary.mean / noise)
        self.theory_mean[code] = torch.where(mask, updated_mean, torch.zeros_like(updated_mean))
        self.theory_var[code] = torch.where(mask, updated_var, torch.zeros_like(updated_var))
        # Primitive posterior receives one soft, visit-level observation.
        p_precision = self.primitive_var.reciprocal() + visit_summary.responsibility / (noise + self.interaction_var)
        p_var = p_precision.reciprocal()
        p_mean = p_var * (self.primitive_mean / self.primitive_var + visit_summary.responsibility * visit_summary.mean / (noise + self.interaction_var))
        self.primitive_mean = torch.where(mask, p_mean, self.primitive_mean)
        self.primitive_var = torch.where(mask, p_var, self.primitive_var)
        self.completed_visits += visit_summary.responsibility
        self.completed_summaries.append(VisitSummary(code, visit_summary.mean.detach().clone(),
                                                     visit_summary.cov.detach().clone(),
                                                     visit_summary.responsibility))

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
            primitive_mean = memory.primitive_mean.detach()
            primitive_var = memory.primitive_var.detach()
            loss = loss + .5 * (3 * prior_var.log() +
                                 ((primitive_mean - self.prior_mean).square() + primitive_var).sum() / prior_var)
            count = count + 3
            for code, theory_mean in memory.theory_mean.items():
                active = TheoryMemory._mask(code).bool()
                theory_var = memory.theory_var[code].detach()[active]
                theory_mean = theory_mean.detach()[active]
                primitive = primitive_mean[active]
                primitive_uncertainty = primitive_var[active]
                loss = loss + .5 * (active.sum() * interaction.log() +
                        ((theory_mean - primitive).square() + theory_var + primitive_uncertainty).sum() / interaction)
                count = count + active.sum()
            for summary in memory.completed_summaries:
                active = TheoryMemory._mask(summary.code).bool()
                prototype = memory.theory_mean.get(summary.code, memory.primitive_mean).detach()[active]
                prototype_var = memory.theory_var.get(summary.code, memory.primitive_var).detach()[active]
                visit_mean = summary.mean.detach()[active]
                visit_var = summary.cov.detach().diag()[active]
                responsibility = summary.responsibility
                loss = loss + .5 * responsibility * (active.sum() * recurrence.log() +
                         ((visit_mean - prototype).square() + visit_var + prototype_var).sum() / recurrence)
                count = count + responsibility * active.sum()
        return loss / count.clamp_min(1)
