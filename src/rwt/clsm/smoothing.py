"""Proposal-corrected finite-mixture and event-aware Gaussian smoothing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .belief import LawComponent, StepAncestry, _chol_with_jitter


@dataclass
class DiscreteSmoothingResult:
    gamma: list[Tensor]
    xi: list[Tensor]


@dataclass
class GaussianSmoothingResult:
    mean: list[list[Tensor]]
    cov: list[list[Tensor]]
    event_xi: list[dict[str, Tensor]]


def backward_component_weights(filtered: list[Tensor], ancestry: list[StepAncestry]) -> DiscreteSmoothingResult:
    """Approximate FFBS discrete weights; exact without pruning/merging.

    `filtered[t]` weights are after emission t. `ancestry[t]` maps those
    components to the pre-emission mixture at t+1. Its columns must match
    `filtered[t+1]` before the emission update.
    """
    if not filtered or len(ancestry) != len(filtered) - 1:
        raise ValueError("need one ancestry transition between consecutive filters")
    gamma: list[Tensor] = [torch.empty(0)] * len(filtered)
    xi: list[Tensor] = [torch.empty(0)] * len(ancestry)
    gamma[-1] = filtered[-1].to(torch.float64) / filtered[-1].sum()
    for t in range(len(ancestry) - 1, -1, -1):
        step = ancestry[t]
        w = filtered[t].to(torch.float64)
        pred = step.predicted_weight.to(torch.float64)
        trans = step.transition.to(torch.float64)
        if trans.shape != (len(w), len(pred)) or gamma[t + 1].shape != pred.shape:
            raise ValueError("ancestry/filter component shape mismatch")
        if torch.any((pred == 0) & (gamma[t + 1] > 0)):
            raise ValueError("smoothed mass on zero predictive support")
        log_ratio = torch.where(pred > 0, gamma[t + 1].log() - pred.log(), -torch.inf)
        log_joint = w.log()[:, None] + trans.log() + log_ratio[None, :]
        normalizer = torch.logsumexp(log_joint.flatten(), 0)
        if not torch.isfinite(normalizer):
            raise RuntimeError("backward normalizer is nonfinite")
        xi[t] = (log_joint - normalizer).exp()
        gamma[t] = xi[t].sum(1)
    return DiscreteSmoothingResult(gamma, xi)


def backward_gaussian_moments(filtered: list[list[LawComponent]],
                              ancestry: list[StepAncestry],
                              discrete: DiscreteSmoothingResult,
                              drift_variance: Tensor,
                              jitter_initial: float = 1e-9,
                              jitter_max: float = 1e-3) -> GaussianSmoothingResult:
    """Event-aware approximate RTS moments on the retained component graph.

    A same-code drift uses its linear-Gaussian backward conditional. Reset and
    structural-entry events are independent of the previous visit parameter
    given the prototype approximation, so their parent conditional is the
    filtered parent. Merged child Gaussians are an explicit approximation.
    """
    if len(filtered) != len(discrete.gamma) or len(ancestry) + 1 != len(filtered):
        raise ValueError("smoother time dimensions mismatch")
    if drift_variance.shape != (3,) or torch.any(drift_variance <= 0):
        raise ValueError("positive three-dimensional drift variance required")
    means: list[list[Tensor]] = [[] for _ in filtered]
    covs: list[list[Tensor]] = [[] for _ in filtered]
    events: list[dict[str, Tensor]] = [{} for _ in ancestry]
    means[-1] = [c.mean.clone() for c in filtered[-1]]
    covs[-1] = [c.cov.clone() for c in filtered[-1]]
    for t in range(len(ancestry) - 1, -1, -1):
        step = ancestry[t]
        if step.event_transition is None:
            raise ValueError("event-specific ancestry required")
        event_xi = {}
        for name, matrix in step.event_transition.items():
            ratio = torch.where(step.transition > 0, matrix / step.transition.clamp_min(1e-300), 0.)
            event_xi[name] = discrete.xi[t] * ratio
        if not torch.allclose(sum(event_xi.values()), discrete.xi[t], atol=1e-10):
            raise RuntimeError("event posterior mass mismatch")
        events[t] = event_xi
        for k, parent in enumerate(filtered[t]):
            mass = discrete.gamma[t][k]
            if float(mass) <= 0:
                means[t].append(parent.mean.clone())
                covs[t].append(parent.cov.clone())
                continue
            conditional = []
            for event, matrix in event_xi.items():
                for j in torch.nonzero(matrix[k] > 0, as_tuple=True)[0].tolist():
                    weight = matrix[k, j] / mass
                    mean = parent.mean.clone()
                    cov = parent.cov.clone()
                    if event == "stay_drift":
                        active = torch.tensor([(parent.code >> dim) & 1 for dim in range(3)], dtype=torch.bool)
                        ids = active.nonzero(as_tuple=True)[0]
                        if len(ids):
                            prior = parent.cov[ids[:,None], ids[None,:]]
                            predicted = prior + torch.diag(drift_variance[ids])
                            chol, _ = _chol_with_jitter(predicted, jitter_initial, jitter_max)
                            gain = torch.cholesky_solve(prior, chol).T
                            child_mean = means[t+1][j][ids]
                            child_cov = covs[t+1][j][ids[:,None], ids[None,:]]
                            mean[ids] = parent.mean[ids] + gain @ (child_mean - parent.mean[ids])
                            block = prior + gain @ (child_cov - predicted) @ gain.T
                            block = (block + block.T) / 2
                            eig, vec = torch.linalg.eigh(block)
                            cov[ids[:,None], ids[None,:]] = (vec * eig.clamp_min(0)) @ vec.T
                    conditional.append((weight, mean, cov))
            if not conditional:
                raise RuntimeError("positive smoothed component has no ancestor event")
            mean = sum((w * m for w,m,_ in conditional), torch.zeros(3,dtype=torch.float64))
            cov = torch.zeros((3,3),dtype=torch.float64)
            for w,m,c in conditional:
                diff = m - mean
                cov += w * (c + torch.outer(diff,diff))
            means[t].append(mean)
            covs[t].append((cov + cov.T) / 2)
    return GaussianSmoothingResult(means, covs, events)
