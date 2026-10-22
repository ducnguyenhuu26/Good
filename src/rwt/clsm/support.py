"""Finite unlabeled actuator-partition hypotheses for CLSM-R2."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class PartitionScore:
    mask: Tensor
    train_log_evidence: float


class DisjointPartition:
    """Canonical unordered bipartitions of actuator channels.

    Group labels are canonicalized by requiring actuator zero in column zero.
    Both groups are nonempty and every selected actuator belongs to exactly
    one group. This class contains no simulator/evaluator imports.
    """

    @staticmethod
    def enumerate(action_dim: int, *, device=None) -> list[Tensor]:
        if not 2 <= action_dim <= 8:
            raise ValueError("partition search supports 2..8 selected actuators")
        result = []
        # Bit zero is fixed into group zero to quotient out label permutation.
        for assignment in range(1 << (action_dim - 1)):
            group0 = [True] + [bool((assignment >> (i - 1)) & 1)
                               for i in range(1, action_dim)]
            if all(group0):
                continue
            group1 = [not value for value in group0]
            result.append(torch.tensor(list(zip(group0, group1)), dtype=torch.bool,
                                       device=device))
        return result

    @staticmethod
    def canonical(mask: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[1] != 2 or mask.shape[0] < 2:
            raise ValueError("partition must have shape [action_dim, 2]")
        if not torch.all(mask.sum(1) == 1) or not torch.all(mask.sum(0) > 0):
            raise ValueError("partition groups must be disjoint, cover channels, and be nonempty")
        if not bool(mask[0, 0]):
            mask = mask.flip(1)
        return mask

    @staticmethod
    def validate(mask: Tensor, action_dim: int | None = None) -> Tensor:
        canonical = DisjointPartition.canonical(mask)
        if action_dim is not None and canonical.shape[0] != action_dim:
            raise ValueError("partition actuator dimension mismatch")
        return canonical

    @staticmethod
    def profile_score(candidate: Tensor, worlds, dynamics, *,
                      gh_nodes: int = 12, mode_prior: dict[int, float] | None = None,
                      batch_size: int = 4096) -> Tensor:
        """Unlabeled stationary-world log evidence for a candidate partition.

        `worlds` contains public `(obs, action, next_obs)` tensors. Each world
        has one constant latent code drawn from the declared design prior and
        one active standard-normal strength for the singleton modes. Simulator
        labels and physical strengths are neither accepted nor used.
        """
        candidate = DisjointPartition.validate(candidate, dynamics.schema.action_dim)
        if gh_nodes < 2 or batch_size < 1 or not worlds:
            raise ValueError("positive stationary worlds, GH nodes>=2, and batch size required")
        prior = mode_prior or {0: .25, 1: .25, 2: .25, 4: .25}
        if set(prior) != {0, 1, 2, 4} or any(p <= 0 for p in prior.values()):
            raise ValueError("calibration design prior must cover 000 and each singleton mode")
        total_prior = sum(prior.values())
        prior = {code: value / total_prior for code, value in prior.items()}
        import numpy as np
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(gh_nodes)
        nodes = torch.as_tensor(nodes_np * math.sqrt(2), dtype=torch.float64)
        log_weights = torch.as_tensor(np.log(weights_np / math.sqrt(math.pi)), dtype=torch.float64)
        log_prior = {code: math.log(p) for code, p in prior.items()}
        dynamics.mechanics.set_partition(candidate)
        world_scores = []
        for world in worlds:
            if len(world) != 3:
                raise ValueError("each calibration world must contain public obs/action/next_obs")
            obs, action, next_obs = (torch.as_tensor(value, dtype=torch.float64,
                                                     device=next(dynamics.parameters()).device)
                                     for value in world)
            if obs.ndim != 2 or action.shape != (len(obs), dynamics.schema.action_dim) or next_obs.shape != obs.shape or len(obs) < 1:
                raise ValueError("malformed stationary calibration world")
            mode_log_evidence = []
            for code in (0, 1, 2, 4):
                latent_log_likelihoods = []
                strength_nodes = torch.zeros((gh_nodes, 3), dtype=obs.dtype, device=obs.device)
                active_coordinate = {1: 0, 2: 1, 4: 2}.get(code)
                if active_coordinate is not None:
                    strength_nodes[:, active_coordinate] = nodes.to(obs.device)
                codes = torch.full((gh_nodes,), code, dtype=torch.long, device=obs.device)
                node_scores = torch.zeros(gh_nodes, dtype=obs.dtype, device=obs.device)
                with torch.no_grad():
                    for start in range(0, len(obs), batch_size):
                        stop = min(start + batch_size, len(obs))
                        count = stop - start
                        for node_start in range(0, gh_nodes, max(1, batch_size // count)):
                            node_stop = min(gh_nodes, node_start + max(1, batch_size // count))
                            ns = node_stop - node_start
                            expanded_obs = obs[start:stop].repeat(ns, 1)
                            expanded_action = action[start:stop].repeat(ns, 1)
                            expanded_next = next_obs[start:stop].repeat(ns, 1)
                            theta = strength_nodes[node_start:node_stop, None, :].expand(ns, count, 3).reshape(ns * count, 3)
                            code_batch = codes[node_start:node_stop, None].expand(ns, count).reshape(-1)
                            mean, variance = dynamics(expanded_obs, expanded_action, code_batch, theta)
                            log_prob = -.5 * (((expanded_next - mean).square() / variance) +
                                              variance.log() + math.log(2 * math.pi)).sum(-1)
                            node_scores[node_start:node_stop] += log_prob.reshape(ns, count).sum(-1)
                if code == 0:
                    marginal = node_scores[0]
                else:
                    marginal = torch.logsumexp(log_weights.to(obs.device) + node_scores, 0)
                mode_log_evidence.append(marginal + log_prior[code])
            world_scores.append(torch.logsumexp(torch.stack(mode_log_evidence), 0))
        return torch.stack(world_scores).sum()

    @staticmethod
    def select_by_profile(dynamics, worlds, *, gh_nodes: int = 12,
                          mode_prior: dict[int, float] | None = None,
                          batch_size: int = 4096) -> list[PartitionScore]:
        scores = []
        for mask in DisjointPartition.enumerate(dynamics.schema.action_dim,
                                                device=dynamics.mechanics.group_mask.device):
            score = DisjointPartition.profile_score(mask, worlds, dynamics,
                                                    gh_nodes=gh_nodes,
                                                    mode_prior=mode_prior,
                                                    batch_size=batch_size)
            scores.append(PartitionScore(mask.detach().cpu().clone(), float(score)))
        scores.sort(key=lambda item: item.train_log_evidence, reverse=True)
        if scores:
            dynamics.mechanics.set_partition(scores[0].mask)
        return scores


def bounded_physical_strength(xi: Tensor,
                              motor_bounds: tuple[float, float] = (0.50, 0.85),
                              damping_bounds: tuple[float, float] = (1.30, 2.00)) -> tuple[Tensor, Tensor]:
    """Map standard Gaussian coordinates into benchmark multiplier bounds."""
    if xi.shape[-1] != 3:
        raise ValueError("strength coordinates must end in three dimensions")
    probability = torch.special.ndtr(xi)
    motor = motor_bounds[0] + (motor_bounds[1] - motor_bounds[0]) * probability[..., :2]
    damping = damping_bounds[0] + (damping_bounds[1] - damping_bounds[0]) * probability[..., 2:3]
    return motor, damping
