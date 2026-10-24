"""Candidate program scoring from old support and adaptation observations only."""
from __future__ import annotations
import torch
from q1_v2.bayes import GaussianPosterior, marginal_nll
from q1_v2.operators import AffineOperatorBank


def composed(bank: AffineOperatorBank, program: tuple[int, ...]):
    result = bank.identity(sum(bank.widths), dtype=bank.bias.dtype, device=bank.bias.device)
    for code in program:
        result = bank.compose(result, (bank.matrix(code), bank.bias[code]))
    return result


def score_program(bank: AffineOperatorBank, old: GaussianPosterior, program: tuple[int, ...],
                  phi_adapt: torch.Tensor, y_adapt: torch.Tensor, sigma2: float,
                  length_penalty: float = 0.0, *, point_estimate: bool = False):
    a, b = composed(bank, program)
    mean = a @ old.mean + b
    covariance = a @ old.covariance @ a.T
    if point_estimate:
        covariance = torch.zeros_like(covariance)
    pred = GaussianPosterior(mean, covariance, old.jitter)
    nll, jitter = marginal_nll(phi_adapt, y_adapt, pred, sigma2)
    return nll + float(length_penalty) * len(program), {"nll": float(nll.detach()), "jitter": jitter,
        "posterior_trace": float(torch.trace(covariance).detach()), "length": len(program)}


def rank_programs(bank, old, programs, phi_adapt, y_adapt, sigma2, length_penalty=0.0,
                  *, point_estimate=False):
    scored = [(program, *score_program(bank, old, program, phi_adapt, y_adapt,
              sigma2, length_penalty, point_estimate=point_estimate)) for program in programs]
    return sorted(scored, key=lambda row: row[1])
