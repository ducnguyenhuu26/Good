"""Exogenous sparse-edit semi-Markov evolution of executable laws."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvolutionBranch:
    code: int
    age: int
    mean: Tensor
    cov: Tensor
    probability: float
    event: str


class LawEvolution(nn.Module):
    def __init__(self, reset_prob: float = 0.05, q_min: float = 1e-6):
        super().__init__()
        if not 1e-4 <= reset_prob <= 0.25:
            raise ValueError("reset probability outside registered range")
        self.log_kappa0 = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        self.log_kappa_pop = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.nu_raw = nn.Parameter(torch.tensor(0.55, dtype=torch.float64))
        self.beta_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.alpha = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.q_raw = nn.Parameter(torch.full((3,), -6.0, dtype=torch.float64))
        self.reset_raw = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(reset_prob))), dtype=torch.float64))
        self.q_min = float(q_min)

    @staticmethod
    def active(code: int) -> Tensor:
        return torch.tensor([(code >> j) & 1 for j in range(3)], dtype=torch.float64)

    @property
    def nu(self) -> Tensor:
        return F.softplus(self.nu_raw) + 1e-8

    @property
    def q(self) -> Tensor:
        return F.softplus(self.q_raw) + self.q_min

    @property
    def reset_prob(self) -> Tensor:
        return self.reset_raw.sigmoid().clamp(1e-4, 0.25)

    def hazard(self, code: Tensor, age: Tensor) -> Tensor:
        if torch.any((code < 0) | (code > 7)) or torch.any(age < 0):
            raise ValueError("invalid structural code or age")
        pop = torch.stack([((code.long() >> j) & 1) for j in range(3)], dim=-1).sum(-1)
        kappa = torch.exp(self.log_kappa0 + self.log_kappa_pop * pop)
        u = age.to(torch.float64)
        # The exact current-age term is zero at u=0. Evaluate the power on a
        # positive base before selecting that value so autograd never forms
        # the undefined derivative of 0**nu when nu is below one.
        current = torch.where(u == 0, torch.zeros_like(u),
                              (u / kappa).clamp_min(1e-12).pow(self.nu))
        rate = ((u + 1) / kappa).pow(self.nu) - current
        return -torch.expm1(-rate)

    def structural_log_transition(self) -> Tensor:
        codes = torch.arange(8, device=self.alpha.device)
        bits = torch.stack([((codes >> j) & 1) for j in range(3)], dim=-1).to(torch.float64)
        dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
        logits = -F.softplus(self.beta_raw) * dist + bits @ self.alpha
        logits = logits.masked_fill(torch.eye(8, dtype=torch.bool, device=logits.device), -torch.inf)
        return logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    def predict_branches(self, component, memory) -> list[EvolutionBranch]:
        """Branch probabilities include stay-drift, stay-reset and seven switches."""
        code, age = int(component.code), int(component.age)
        h = float(self.hazard(torch.tensor([code]), torch.tensor([age]))[0].detach())
        rho = float(self.reset_prob.detach())
        mask = self.active(code).to(component.mean.device)
        drift_cov = component.cov + torch.diag(self.q.detach().to(mask.device) * mask)
        branches = [EvolutionBranch(code, age + 1, component.mean.clone(), drift_cov,
                                    (1 - h) * (1 - rho), "stay_drift")]
        entry_mean, entry_cov = memory.prior_for_entry(code, component.scratch_or_reuse)
        branches.append(EvolutionBranch(code, age + 1, entry_mean, entry_cov,
                                        (1 - h) * rho, "stay_reset"))
        probabilities = self.structural_log_transition()[code].exp().detach()
        for target in range(8):
            if target == code:
                continue
            mean, cov = memory.prior_for_entry(target, component.scratch_or_reuse)
            branches.append(EvolutionBranch(target, 0, mean, cov,
                                            h * float(probabilities[target]), "switch"))
        return branches


def forward_log_evidence_discrete(log_emissions: Tensor, initial: Tensor, transition: Tensor) -> Tensor:
    """Exact evidence: emit at t under current law, then transition for t+1.

    log_emissions [T,S], initial [S], transition [S,S] row stochastic.
    """
    if log_emissions.ndim != 2 or initial.shape != (log_emissions.shape[1],) or transition.shape != (len(initial), len(initial)):
        raise ValueError("discrete forward shapes mismatch")
    if not torch.allclose(initial.sum(), torch.tensor(1.0, dtype=initial.dtype), atol=1e-12):
        raise ValueError("initial state mass must sum to one")
    alpha = initial.to(torch.float64).log()
    log_t = transition.to(torch.float64).log()
    for t, emission in enumerate(log_emissions.to(torch.float64)):
        alpha = alpha + emission
        if t + 1 < len(log_emissions):
            alpha = torch.logsumexp(alpha[:, None] + log_t, dim=0)
    return torch.logsumexp(alpha, dim=0)
