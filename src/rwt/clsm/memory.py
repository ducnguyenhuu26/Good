"""Causal Gaussian primitive/theory memory and prequential scratch comparator."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


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
        if not 0 < scratch_weight < 1 or min(prior_std, interaction_var, recurrence_var) <= 0:
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
        self.completed_visits = 0

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
        self.completed_visits += 1

    def update_prequential_evidence(self, scratch_log_predictive: float, reuse_log_predictive: float) -> None:
        """Call once *after* the transition whose predictive densities these are."""
        self.log_model_weights += torch.tensor([scratch_log_predictive, reuse_log_predictive], dtype=torch.float64)
        self.log_model_weights -= torch.logsumexp(self.log_model_weights, dim=0)

    def scratch_copy(self) -> "TheoryMemory":
        return TheoryMemory(self.prior_mean, self.prior_var ** 0.5,
                            self.interaction_var, self.recurrence_var,
                            float(self.log_model_weights[0].exp()))
